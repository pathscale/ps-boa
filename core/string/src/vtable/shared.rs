//! Arc-backed [`JsString`] kind.
//!
//! [`SharedJsStr`] owns Latin1 or UTF-16 code units in a buffer held by an
//! [`Arc`][`std::sync::Arc`]. The value is [`Send`] and [`Sync`].
//! [`JsString::from_shared`] turns one into a [`JsString`] without copying the
//! units. Cloning and dropping that `JsString` adjust the same atomic count.

use crate::iter::CodePointsIter;
use crate::r#type::{InternalStringType, Latin1, Utf16};
use crate::vtable::JsStringVTable;
use crate::{JsStr, JsStrVariant, JsString, JsStringKind};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, offset_of};
use std::ptr::NonNull;
use std::sync::Arc;

// Distinct from `InternalStringType::KIND`, which selects the non-atomic sequence kind.
trait SharedType: InternalStringType {
    const SHARED_KIND: JsStringKind;
}

impl SharedType for Latin1 {
    const SHARED_KIND: JsStringKind = JsStringKind::Latin1Shared;
}

impl SharedType for Utf16 {
    const SHARED_KIND: JsStringKind = JsStringKind::Utf16Shared;
}

/// Header stored inside the [`Arc`]. The vtable is the first field so a [`JsString`]
/// can address this allocation directly.
///
/// The code units are a [`Box`], not a trailing slice. [`Arc`] has no public
/// constructor for a custom dynamically sized header, and the vtable pointer has
/// to be the `Arc`'s `T` so clone and drop can use [`Arc::increment_strong_count`]
/// and [`Arc::from_raw`].
#[repr(C)]
struct SharedString<T: SharedType> {
    vtable: JsStringVTable,
    data: Box<[T::Byte]>,
    _marker: PhantomData<fn() -> T>,
}

static_assertions::const_assert!(offset_of!(SharedString<Latin1>, vtable) == 0);
static_assertions::const_assert!(offset_of!(SharedString<Utf16>, vtable) == 0);

#[derive(Clone)]
enum SharedInner {
    Latin1(Arc<SharedString<Latin1>>),
    Utf16(Arc<SharedString<Utf16>>),
}

/// Immutable Latin1 or UTF-16 string stored in an [`Arc`][`std::sync::Arc`].
///
/// Cloning bumps the atomic reference count and does not copy the code units.
/// [`JsString::from_shared`] materializes a [`JsString`] that shares the allocation.
/// Send [`SharedJsStr`] across threads, then call `from_shared` on the thread that
/// will use the [`JsString`]. [`JsString`] itself is not thread-safe.
#[derive(Clone)]
pub struct SharedJsStr {
    inner: SharedInner,
}

const _: fn() = || {
    fn assert<T: Send + Sync>() {}
    assert::<SharedJsStr>();
};

impl SharedJsStr {
    /// Borrows the code units. Does not copy.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> JsStr<'_> {
        match &self.inner {
            SharedInner::Latin1(arc) => JsStr::latin1(&arc.data),
            SharedInner::Utf16(arc) => JsStr::utf16(&arc.data),
        }
    }

    #[inline]
    fn from_latin1(bytes: &[u8]) -> Self {
        Self {
            inner: SharedInner::Latin1(SharedString::<Latin1>::from_units(bytes)),
        }
    }

    #[inline]
    fn from_utf16(units: &[u16]) -> Self {
        Self {
            inner: SharedInner::Utf16(SharedString::<Utf16>::from_units(units)),
        }
    }

    /// Hands this owner to a [`JsString`] without copying the units.
    #[inline]
    #[must_use]
    pub(crate) fn into_js_string(self) -> JsString {
        let ptr = match self.inner {
            SharedInner::Latin1(arc) => vtable_ptr(arc),
            SharedInner::Utf16(arc) => vtable_ptr(arc),
        };
        // SAFETY: `ptr` addresses the vtable of an `Arc` whose strong count this
        // `JsString` now owns. Later clones and drops go through that vtable.
        unsafe { JsString::from_ptr(ptr) }
    }
}

impl<T: SharedType> SharedString<T> {
    fn from_units(units: &[T::Byte]) -> Arc<Self> {
        let data: Box<[T::Byte]> = Box::from(units);
        let len = data.len();
        Arc::new(Self {
            vtable: JsStringVTable {
                clone: shared_clone::<T>,
                drop: shared_drop::<T>,
                as_str: shared_as_str::<T>,
                code_points: shared_code_points::<T>,
                refcount: shared_refcount::<T>,
                len,
                kind: T::SHARED_KIND,
            },
            data,
            _marker: PhantomData,
        })
    }
}

#[inline]
fn shared_clone<T: SharedType>(vtable: NonNull<JsStringVTable>) -> JsString {
    // SAFETY: `vtable` points at a live `SharedString<T>` created by `Arc::into_raw`
    // (or a later `increment_strong_count` on that pointer). The caller holds one
    // strong reference, so the allocation is alive. The increment records the clone.
    // `from_ptr` only copies the address.
    unsafe {
        Arc::increment_strong_count(vtable.cast::<SharedString<T>>().as_ptr());
        JsString::from_ptr(vtable)
    }
}

#[inline]
fn shared_drop<T: SharedType>(vtable: NonNull<JsStringVTable>) {
    // SAFETY: `vtable` was produced by `Arc::into_raw` or by `shared_clone`.
    // Rebuilding the `Arc` and dropping it releases exactly one strong reference
    // and frees the allocation when that was the last owner.
    unsafe {
        drop(Arc::from_raw(vtable.cast::<SharedString<T>>().as_ptr()));
    }
}

#[inline]
fn shared_as_str<T: SharedType>(vtable: NonNull<JsStringVTable>) -> JsStr<'static> {
    // SAFETY: `vtable` is the first field of a live `SharedString<T>` inside an `Arc`.
    // The `JsString` borrow that called this keeps that `Arc` alive.
    let this: &SharedString<T> = unsafe { vtable.cast().as_ref() };
    debug_assert_eq!(this.vtable.len, this.data.len());
    let ptr = this.data.as_ptr();
    let len = this.data.len();
    // SAFETY: `data` is a `Box<[T::Byte]>` of `len` initialized elements, aligned for
    // `T::Byte`. An empty box still returns a non-null aligned pointer. The `Arc`
    // owns the buffer for as long as any `JsString` of this kind is borrowed.
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    T::str_ctor(slice)
}

#[inline]
fn shared_code_points<T: SharedType>(vtable: NonNull<JsStringVTable>) -> CodePointsIter<'static> {
    CodePointsIter::new(shared_as_str::<T>(vtable))
}

/// `VTable` function for refcount, need to return an `Option<usize>`.
#[inline]
#[allow(clippy::unnecessary_wraps)]
fn shared_refcount<T: SharedType>(vtable: NonNull<JsStringVTable>) -> Option<usize> {
    // SAFETY: same live `Arc` as the other vtable functions. `ManuallyDrop` prevents
    // the temporary `Arc` from decrementing the strong count.
    let arc =
        unsafe { ManuallyDrop::new(Arc::from_raw(vtable.cast::<SharedString<T>>().as_ptr())) };
    Some(Arc::strong_count(&arc))
}

/// Rebuild an [`Arc`] for a `JsString` whose kind was already checked.
#[inline]
fn clone_arc<T: SharedType>(string: &JsString) -> Arc<SharedString<T>> {
    debug_assert_eq!(string.kind(), T::SHARED_KIND);
    let ptr = string.ptr.cast::<SharedString<T>>().as_ptr();
    // SAFETY: `string.kind()` selected `T`, and that kind is only built from
    // `Arc::<SharedString<T>>::into_raw`. `string` keeps one strong reference.
    // The increment balances `from_raw`, so `string` keeps its original count.
    unsafe {
        Arc::increment_strong_count(ptr);
        Arc::from_raw(ptr)
    }
}

#[inline]
fn vtable_ptr<T: SharedType>(arc: Arc<SharedString<T>>) -> NonNull<JsStringVTable> {
    let raw = Arc::into_raw(arc);
    // SAFETY: `Arc::into_raw` never returns null. `#[repr(C)]` places `vtable` at
    // offset 0, so this is a valid `JsStringVTable` pointer. The strong count is
    // transferred, not incremented.
    unsafe { NonNull::new_unchecked(raw.cast_mut()).cast() }
}

impl From<&JsString> for SharedJsStr {
    #[inline]
    fn from(value: &JsString) -> Self {
        match value.kind() {
            JsStringKind::Latin1Shared => Self {
                inner: SharedInner::Latin1(clone_arc::<Latin1>(value)),
            },
            JsStringKind::Utf16Shared => Self {
                inner: SharedInner::Utf16(clone_arc::<Utf16>(value)),
            },
            _ => match value.as_str().variant() {
                JsStrVariant::Latin1(bytes) => Self::from_latin1(bytes),
                JsStrVariant::Utf16(units) => Self::from_utf16(units),
            },
        }
    }
}

impl From<&str> for SharedJsStr {
    #[inline]
    fn from(s: &str) -> Self {
        // Same Latin1 / UTF-16 choice as `From<&str> for JsString`, without interning.
        // `PartialEq<str>` compares Latin1 against raw bytes, so the cutoff has to match.
        if s.is_ascii() {
            return Self::from_latin1(s.as_bytes());
        }
        if s.chars().all(|c| c as u32 <= 0xFF) {
            let bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
            return Self::from_latin1(&bytes);
        }
        let units = s.encode_utf16().collect::<Vec<_>>();
        Self::from_utf16(&units)
    }
}

impl From<&[u16]> for SharedJsStr {
    #[inline]
    fn from(units: &[u16]) -> Self {
        Self::from_utf16(units)
    }
}

impl<const N: usize> From<&[u16; N]> for SharedJsStr {
    #[inline]
    fn from(units: &[u16; N]) -> Self {
        Self::from(&units[..])
    }
}

impl PartialEq for SharedJsStr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SharedJsStr {}

impl PartialOrd for SharedJsStr {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SharedJsStr {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(&other.as_str())
    }
}

impl Hash for SharedJsStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl PartialEq<JsString> for SharedJsStr {
    #[inline]
    fn eq(&self, other: &JsString) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<SharedJsStr> for JsString {
    #[inline]
    fn eq(&self, other: &SharedJsStr) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<str> for SharedJsStr {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for SharedJsStr {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<SharedJsStr> for str {
    #[inline]
    fn eq(&self, other: &SharedJsStr) -> bool {
        other == self
    }
}

impl std::fmt::Debug for SharedJsStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let units = self.as_str();
        let content = units.display_lossy().to_string();
        f.debug_struct("SharedJsStr")
            .field("len", &units.len())
            .field("latin1", &units.is_latin1())
            .field("content", &content)
            .finish_non_exhaustive()
    }
}
