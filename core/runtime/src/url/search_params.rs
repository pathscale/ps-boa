//! Boa's implementation of JavaScript's `URLSearchParams` Web API class.
//!
//! More information:
//!  - [MDN documentation][mdn]
//!  - [WHATWG `URL` specification][spec]
//!
//! [spec]: https://url.spec.whatwg.org/#interface-urlsearchparams
//! [mdn]: https://developer.mozilla.org/en-US/docs/Web/API/URLSearchParams
#![allow(clippy::needless_pass_by_value)]

use boa_engine::interop::JsClass;
use boa_engine::object::builtins::TypedJsFunction;
use boa_engine::value::{Convert, TryFromJs};
use boa_engine::{
    Context, Finalize, JsData, JsObject, JsResult, JsString, JsValue, Trace, boa_class, js_error,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use super::search_params_iterator::{IterationKind, UrlSearchParamsIterator};

/// The callback `forEach` is given: `(value, name, searchParams)`.
pub type ForEachCallback = TypedJsFunction<(JsString, JsString, JsObject), ()>;

/// Where a `URLSearchParams` keeps its pairs.
///
/// The two cases are the two ways the specification says one comes into being,
/// and they behave differently on purpose. A `URLSearchParams` constructed
/// directly owns its list. One handed out by `URL.searchParams` is a *view*:
/// the specification ties it to the URL, so reading it must see writes made
/// through `url.search`, and writing it must change the URL. Holding the same
/// `Rc<RefCell<url::Url>>` the `URL` object holds is what makes both true
/// without a synchronisation step that could be skipped.
#[derive(Debug, Clone)]
pub(crate) enum Backing {
    /// Its own list, for `new URLSearchParams(...)`.
    Detached(Rc<RefCell<Vec<(String, String)>>>),
    /// A view over a URL's query, for `URL.searchParams`.
    Linked(Rc<RefCell<url::Url>>),
}

/// A JavaScript wrapper for the `URLSearchParams` object.
#[derive(Debug, Clone, JsData, Trace, Finalize)]
pub struct UrlSearchParams {
    // Neither case holds a `JsValue`, so there is nothing here for the
    // collector to trace.
    #[unsafe_ignore_trace]
    pub(crate) backing: Backing,
}

impl Default for UrlSearchParams {
    fn default() -> Self {
        Self {
            backing: Backing::Detached(Rc::new(RefCell::new(Vec::new()))),
        }
    }
}

impl UrlSearchParams {
    /// Creates a view over a URL's query string.
    #[must_use]
    pub(crate) fn linked(url: Rc<RefCell<url::Url>>) -> Self {
        Self {
            backing: Backing::Linked(url),
        }
    }

    /// The current list, decoded.
    #[must_use]
    pub(crate) fn pairs(&self) -> Vec<(String, String)> {
        match &self.backing {
            Backing::Detached(pairs) => pairs.borrow().clone(),
            Backing::Linked(url) => url
                .borrow()
                .query_pairs()
                .map(|(name, value)| (name.into_owned(), value.into_owned()))
                .collect(),
        }
    }

    /// Replaces the list.
    fn set_pairs(&self, pairs: Vec<(String, String)>) {
        match &self.backing {
            Backing::Detached(own) => *own.borrow_mut() = pairs,
            Backing::Linked(url) => {
                let mut url = url.borrow_mut();
                if pairs.is_empty() {
                    // Not `query_pairs_mut().clear()`, which leaves a bare `?`
                    // behind. An empty list means the URL has no query at all,
                    // and `url.search` has to come back as the empty string.
                    url.set_query(None);
                } else {
                    let mut serializer = url.query_pairs_mut();
                    serializer.clear();
                    for (name, value) in &pairs {
                        serializer.append_pair(name, value);
                    }
                    serializer.finish();
                }
            }
        }
    }

    /// Parses an `init` string, with or without a leading `?`.
    fn from_query_string(init: &str) -> Vec<(String, String)> {
        let trimmed = init.strip_prefix('?').unwrap_or(init);
        url::form_urlencoded::parse(trimmed.as_bytes())
            .map(|(name, value)| (name.into_owned(), value.into_owned()))
            .collect()
    }
}

impl TryFromJs for UrlSearchParams {
    fn try_from_js(value: &JsValue, context: &mut Context) -> JsResult<Self> {
        UrlSearchParams::constructor(Some(value.clone()), context)
    }
}

#[boa_class(rename = "URLSearchParams")]
#[boa(rename_all = "camelCase")]
impl UrlSearchParams {
    /// Creates a `URLSearchParams`.
    ///
    /// `init` may be a query string, a sequence of two-element pairs, a record,
    /// or another `URLSearchParams`.
    ///
    /// # Errors
    /// If `init` is none of those shapes.
    #[boa(constructor)]
    pub fn constructor(init: Option<JsValue>, context: &mut Context) -> JsResult<Self> {
        let params = Self::default();

        let Some(init) = init else {
            return Ok(params);
        };
        if init.is_undefined() || init.is_null() {
            return Ok(params);
        }

        // Another `URLSearchParams` copies its list, and copies it *now*: the
        // specification says the new object is independent, so a view over a
        // URL must not stay a view once it has been copied out of one.
        if let Some(other) = init
            .as_object()
            .as_ref()
            .and_then(JsObject::downcast_ref::<UrlSearchParams>)
        {
            params.set_pairs(other.pairs());
            return Ok(params);
        }

        if let Some(string) = init.as_string() {
            params.set_pairs(Self::from_query_string(&string.to_std_string_lossy()));
            return Ok(params);
        }

        // A sequence is tried before a record, because an array is also an
        // object and would otherwise be read as one with numeric keys.
        if let Ok(pairs) = Vec::<(Convert<String>, Convert<String>)>::try_from_js(&init, context) {
            params.set_pairs(
                pairs
                    .into_iter()
                    .map(|(k, v)| (k.0.clone(), v.0.clone()))
                    .collect(),
            );
            return Ok(params);
        }

        if let Ok(record) = BTreeMap::<String, Convert<String>>::try_from_js(&init, context) {
            params.set_pairs(record.into_iter().map(|(k, v)| (k, v.0.clone())).collect());
            return Ok(params);
        }

        Err(js_error!(TypeError: "Cannot convert init to URLSearchParams."))
    }

    /// Appends a name/value pair, keeping any pair already stored under that
    /// name.
    pub fn append(&self, name: Convert<String>, value: Convert<String>) {
        let mut pairs = self.pairs();
        pairs.push((name.0.clone(), value.0.clone()));
        self.set_pairs(pairs);
    }

    /// Removes pairs by name, or by name and value when a value is given.
    pub fn delete(&self, name: Convert<String>, value: Option<Convert<String>>) {
        let pairs = self
            .pairs()
            .into_iter()
            .filter(|(existing_name, existing_value)| match &value {
                Some(value) => !(existing_name == &name.0 && existing_value == &value.0),
                None => existing_name != &name.0,
            })
            .collect();
        self.set_pairs(pairs);
    }

    /// The first value stored under `name`, or `null`.
    #[must_use]
    pub fn get(&self, name: Convert<String>) -> JsValue {
        self.pairs()
            .into_iter()
            .find(|(existing, _)| existing == &name.0)
            .map_or_else(JsValue::null, |(_, value)| {
                JsValue::from(JsString::from(value))
            })
    }

    /// Every value stored under `name`, in order.
    #[boa(method)]
    #[must_use]
    pub fn get_all(&self, name: Convert<String>, context: &mut Context) -> JsValue {
        let values = self
            .pairs()
            .into_iter()
            .filter(|(existing, _)| existing == &name.0)
            .map(|(_, value)| JsValue::from(JsString::from(value)));
        boa_engine::object::builtins::JsArray::from_iter(values, context).into()
    }

    /// Whether a pair is stored under `name`, or under `name` with `value`.
    #[must_use]
    pub fn has(&self, name: Convert<String>, value: Option<Convert<String>>) -> bool {
        self.pairs()
            .into_iter()
            .any(|(existing_name, existing_value)| match &value {
                Some(value) => existing_name == name.0 && existing_value == value.0,
                None => existing_name == name.0,
            })
    }

    /// Sets `name` to `value`, replacing the first pair with that name and
    /// dropping the rest.
    pub fn set(&self, name: Convert<String>, value: Convert<String>) {
        let mut replaced = false;
        let mut pairs = Vec::new();
        for (existing_name, existing_value) in self.pairs() {
            if existing_name == name.0 {
                if !replaced {
                    replaced = true;
                    pairs.push((name.0.clone(), value.0.clone()));
                }
            } else {
                pairs.push((existing_name, existing_value));
            }
        }
        if !replaced {
            pairs.push((name.0.clone(), value.0.clone()));
        }
        self.set_pairs(pairs);
    }

    /// Sorts the pairs by name, keeping the relative order of pairs that share
    /// one.
    pub fn sort(&self) {
        let mut pairs = self.pairs();
        pairs.sort_by(|(left, _), (right, _)| left.cmp(right));
        self.set_pairs(pairs);
    }

    /// The number of pairs.
    #[boa(getter)]
    #[must_use]
    pub fn size(&self) -> usize {
        self.pairs().len()
    }

    /// The list as a query string, without a leading `?`.
    #[boa(method)]
    #[must_use]
    pub fn to_string(&self) -> JsString {
        let mut serializer = url::form_urlencoded::Serializer::new(String::new());
        for (name, value) in self.pairs() {
            serializer.append_pair(&name, &value);
        }
        JsString::from(serializer.finish())
    }

    /// Runs `callback` once per pair, as `(value, name, searchParams)`.
    ///
    /// # Errors
    /// Whatever the callback throws.
    #[boa(method)]
    pub fn for_each(
        this: JsClass<Self>,
        callback: ForEachCallback,
        this_arg: Option<JsValue>,
        context: &mut Context,
    ) -> JsResult<()> {
        let object = this.inner().upcast();
        let this_arg = this_arg.unwrap_or_default();
        for (name, value) in this.clone_inner().pairs() {
            callback.call_with_this(
                &this_arg,
                context,
                (JsString::from(value), JsString::from(name), object.clone()),
            )?;
        }
        Ok(())
    }

    /// An iterator over the name/value pairs.
    ///
    /// # Errors
    /// If the iterator class is not registered in this realm.
    #[boa(method)]
    pub fn entries(this: JsClass<Self>, context: &mut Context) -> JsResult<JsValue> {
        UrlSearchParamsIterator::create(this.inner(), IterationKind::KeyAndValue, context)
    }

    /// An iterator over the names.
    ///
    /// # Errors
    /// If the iterator class is not registered in this realm.
    #[boa(method)]
    pub fn keys(this: JsClass<Self>, context: &mut Context) -> JsResult<JsValue> {
        UrlSearchParamsIterator::create(this.inner(), IterationKind::Key, context)
    }

    /// An iterator over the values.
    ///
    /// # Errors
    /// If the iterator class is not registered in this realm.
    #[boa(method)]
    pub fn values(this: JsClass<Self>, context: &mut Context) -> JsResult<JsValue> {
        UrlSearchParamsIterator::create(this.inner(), IterationKind::Value, context)
    }

    /// `for...of` over a `URLSearchParams` walks its pairs.
    ///
    /// # Errors
    /// If the iterator class is not registered in this realm.
    #[boa(method)]
    #[boa(symbol = "iterator")]
    pub fn symbol_iterator(this: JsClass<Self>, context: &mut Context) -> JsResult<JsValue> {
        UrlSearchParamsIterator::create(this.inner(), IterationKind::KeyAndValue, context)
    }
}
