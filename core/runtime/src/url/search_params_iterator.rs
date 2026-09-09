//! This module implements the `URLSearchParams Iterator` object.
//!
//! More information:
//!  - [WHATWG `URL` specification][spec]
//!
//! [spec]: https://url.spec.whatwg.org/#interface-urlsearchparams

use boa_engine::{
    Context, JsData, JsResult, JsString, JsValue, boa_class,
    builtins::iterable::create_iter_result_object, error::JsNativeError, interop::JsClass,
    object::JsObject, object::builtins::JsArray,
};
use boa_gc::{Finalize, Trace};

use super::search_params::UrlSearchParams;

/// Which half of each pair the iterator yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IterationKind {
    /// Iterates over the names.
    Key,
    /// Iterates over the values.
    Value,
    /// Iterates over the names and values, as two-element arrays.
    KeyAndValue,
}

/// An iteration over a `URLSearchParams` object, implementing the iterator
/// protocol.
///
/// The list is read afresh on every `next`, rather than snapshotted at
/// creation. That is what the specification asks for -- the iterator walks the
/// live list -- and it matters here because a `URLSearchParams` handed out by
/// `URL.searchParams` is a view over a URL that other code can still write to.
#[derive(Debug, Finalize, Trace, JsData)]
pub(crate) struct UrlSearchParamsIterator {
    iterated_params: JsObject<UrlSearchParams>,
    next_index: usize,
    #[unsafe_ignore_trace]
    iteration_kind: IterationKind,
}

#[boa_class(rename = "URLSearchParams Iterator")]
impl UrlSearchParamsIterator {
    /// Prevent direct construction: instances come from `entries`, `keys`,
    /// `values` and `Symbol.iterator` only.
    #[boa(constructor)]
    fn constructor() -> JsResult<Self> {
        Err(JsNativeError::typ()
            .with_message("Illegal constructor")
            .into())
    }

    /// `%URLSearchParamsIteratorPrototype%.next()`
    #[boa(method)]
    fn next(&mut self, context: &mut Context) -> JsValue {
        let element = self
            .iterated_params
            .borrow()
            .data()
            .pairs()
            .into_iter()
            .nth(self.next_index)
            .map(|(name, value)| {
                (
                    JsValue::from(JsString::from(name)),
                    JsValue::from(JsString::from(value)),
                )
            });

        if let Some((name, value)) = element {
            self.next_index += 1;

            return match self.iteration_kind {
                IterationKind::Key => create_iter_result_object(name, false, context),
                IterationKind::Value => create_iter_result_object(value, false, context),
                IterationKind::KeyAndValue => {
                    let result = JsArray::from_iter([name, value], context);
                    create_iter_result_object(result.into(), false, context)
                }
            };
        }

        create_iter_result_object(JsValue::undefined(), true, context)
    }

    /// Returns `this`, so the iterator is itself iterable and works in
    /// `for...of`.
    #[boa(method)]
    #[boa(symbol = "iterator")]
    fn symbol_iterator(this: JsClass<Self>) -> JsValue {
        this.inner().into()
    }
}

impl UrlSearchParamsIterator {
    /// Creates a new iterator over the given `URLSearchParams` object.
    pub(crate) fn create(
        params: JsObject<UrlSearchParams>,
        kind: IterationKind,
        context: &mut Context,
    ) -> JsResult<JsValue> {
        let iter = Self {
            iterated_params: params,
            next_index: 0,
            iteration_kind: kind,
        };

        let proto = context
            .realm()
            .get_class::<Self>()
            .ok_or_else(|| boa_engine::js_error!(Error: "URLSearchParams Iterator not registered"))?
            .prototype();

        Ok(JsObject::from_proto_and_data(proto, iter).into())
    }
}
