//! This module implements the `ArrayIterator` object.
//!
//! More information:
//!  - [ECMAScript reference][spec]
//!
//! [spec]: https://tc39.es/ecma262/#sec-array-iterator-objects

use crate::{
    Context, JsData, JsResult,
    builtins::{
        Array, BuiltInBuilder, IntrinsicObject, JsValue,
        iterable::{IteratorRecord, create_iter_result_object},
        typed_array::TypedArray,
    },
    context::intrinsics::Intrinsics,
    error::JsNativeError,
    js_string,
    native_function::NativeFunctionObject,
    object::JsObject,
    property::{Attribute, PropertyKey, PropertyNameKind},
    realm::Realm,
    symbol::JsSymbol,
};
use boa_gc::{Finalize, Trace};

/// The Array Iterator object represents an iteration over an array. It implements the iterator protocol.
///
/// More information:
///  - [ECMAScript reference][spec]
///
/// [spec]: https://tc39.es/ecma262/#sec-array-iterator-objects
#[derive(Debug, Clone, Finalize, Trace, JsData)]
pub(crate) struct ArrayIterator {
    array: JsObject,
    next_index: u64,
    #[unsafe_ignore_trace]
    kind: PropertyNameKind,
    done: bool,
}

impl IntrinsicObject for ArrayIterator {
    fn init(realm: &Realm) {
        BuiltInBuilder::with_intrinsic::<Self>(realm)
            .prototype(realm.intrinsics().constructors().iterator().prototype())
            .static_method(Self::next, js_string!("next"), 0)
            .static_property(
                JsSymbol::to_string_tag(),
                js_string!("Array Iterator"),
                Attribute::CONFIGURABLE,
            )
            .build();
    }

    fn get(intrinsics: &Intrinsics) -> JsObject {
        intrinsics.objects().iterator_prototypes().array()
    }
}

impl ArrayIterator {
    fn new(array: JsObject, kind: PropertyNameKind) -> Self {
        Self {
            array,
            kind,
            next_index: 0,
            done: false,
        }
    }

    /// `CreateArrayIterator( array, kind )`
    ///
    /// Creates a new iterator over the given array.
    ///
    /// More information:
    ///  - [ECMA reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-createarrayiterator
    pub(crate) fn create_array_iterator(
        array: JsObject,
        kind: PropertyNameKind,
        context: &Context,
    ) -> JsValue {
        let array_iterator = JsObject::from_proto_and_data_with_shared_shape(
            context.root_shape(),
            context.intrinsics().objects().iterator_prototypes().array(),
            Self::new(array, kind),
        );
        array_iterator.into()
    }

    /// %ArrayIteratorPrototype%.next( )
    ///
    /// Gets the next result in the array.
    ///
    /// More information:
    ///  - [ECMA reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-%arrayiteratorprototype%.next
    pub(crate) fn next(this: &JsValue, _: &[JsValue], context: &mut Context) -> JsResult<JsValue> {
        let (value, done) = Self::next_value(this, context)?;
        Ok(create_iter_result_object(value, done, context))
    }

    /// `GetIterator(o, sync)` for an array still on the array template shape (so no own
    /// `@@iterator`, and `Array.prototype` as its prototype) while `Array.prototype[@@iterator]`
    /// is the intrinsic `values` and `%ArrayIteratorPrototype%.next` the intrinsic `next`, both
    /// data properties: the iterator record those lookups and the `values` call would produce.
    /// `None` when any of that does not hold.
    pub(crate) fn fast_iterator_record(o: &JsObject, context: &Context) -> Option<IteratorRecord> {
        let intrinsics = context.intrinsics();
        if !o.is_array()
            || o.borrow().properties().shape.to_addr_usize()
                != intrinsics.templates().array().shape().to_addr_usize()
        {
            return None;
        }

        let own_data = |object: &JsObject, key: PropertyKey| {
            let object = object.borrow();
            let slot = object.properties().shape.lookup(&key)?;
            if slot.attributes.is_accessor_descriptor() {
                return None;
            }
            Some(object.properties().storage[slot.index as usize].clone())
        };

        let values = own_data(
            &intrinsics.constructors().array().prototype(),
            JsSymbol::iterator().into(),
        )?;
        if values.as_object()? != JsObject::from(intrinsics.objects().array_prototype_values()) {
            return None;
        }
        let next = own_data(
            &intrinsics.objects().iterator_prototypes().array(),
            js_string!("next").into(),
        )?;
        if !Self::is_intrinsic_next(&next) {
            return None;
        }

        let iterator = Self::create_array_iterator(o.clone(), PropertyNameKind::Value, context);
        Some(IteratorRecord::new(iterator.as_object()?, next))
    }

    /// Returns `true` if `next_method` is the intrinsic `%ArrayIteratorPrototype%.next`.
    pub(crate) fn is_intrinsic_next(next_method: &JsValue) -> bool {
        next_method.as_object().is_some_and(|f| {
            f.downcast_ref::<NativeFunctionObject>()
                .is_some_and(|f| f.f.is_fn_ptr(Self::next))
        })
    }

    /// The steps of `%ArrayIteratorPrototype%.next`, returning the value and `done` flag instead of
    /// a result object, so an iterator record over the intrinsic `next` can skip the allocation.
    pub(crate) fn next_value(this: &JsValue, context: &mut Context) -> JsResult<(JsValue, bool)> {
        let object = this.as_object();
        let mut array_iterator = object
            .as_ref()
            .and_then(JsObject::downcast_mut::<Self>)
            .ok_or_else(|| JsNativeError::typ().with_message("`this` is not an ArrayIterator"))?;
        let index = array_iterator.next_index;
        if array_iterator.done {
            return Ok((JsValue::undefined(), true));
        }

        let len = if let Some(f) = array_iterator.array.downcast_ref::<TypedArray>() {
            let buf = f.viewed_array_buffer().as_buffer();
            let Some(buf) = buf
                .bytes(std::sync::atomic::Ordering::SeqCst)
                .filter(|buf| !f.is_out_of_bounds(buf.len()))
            else {
                return Err(JsNativeError::typ()
                    .with_message("Cannot get value from out of bounds typed array")
                    .into());
            };

            f.array_length(buf.len())
        } else {
            array_iterator.array.length_of_array_like(context)?
        };

        if index >= len {
            array_iterator.done = true;
            return Ok((JsValue::undefined(), true));
        }
        array_iterator.next_index = index + 1;
        match array_iterator.kind {
            PropertyNameKind::Key => Ok((index.into(), false)),
            PropertyNameKind::Value => {
                let element_value = array_iterator.array.get(index, context)?;
                Ok((element_value, false))
            }
            PropertyNameKind::KeyAndValue => {
                let element_value = array_iterator.array.get(index, context)?;
                let result = Array::create_array_from_list([index.into(), element_value], context);
                Ok((result.into(), false))
            }
        }
    }
}
