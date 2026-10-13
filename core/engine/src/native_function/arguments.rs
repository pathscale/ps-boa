use arrayvec::ArrayVec;

use crate::{JsValue, vm::Stack};

// The inline variant is the point: it lives on the native call frame so
// calls with up to eight arguments never allocate.
#[allow(variant_size_differences)]
enum Arguments {
    Few(ArrayVec<JsValue, 8>),
    Many(Vec<JsValue>),
}

/// Owns a native call's values independently of the VM stack.
///
/// Re-entry may grow or truncate that stack, so a native argument slice must
/// refer to this local storage instead. Small calls do not allocate.
pub(super) struct NativeCall {
    arguments: Arguments,
    function: JsValue,
    this: JsValue,
}

impl NativeCall {
    pub(super) fn new(stack: &mut Stack, argument_count: usize) -> Self {
        let mut few = ArrayVec::<JsValue, 8>::new();
        let arguments = if argument_count <= few.capacity() {
            stack.calling_convention_pop_arguments_into(argument_count, &mut few);
            Arguments::Few(few)
        } else {
            Arguments::Many(stack.calling_convention_pop_arguments(argument_count))
        };
        let function = stack.pop();
        let this = stack.pop();
        Self {
            arguments,
            function,
            this,
        }
    }

    pub(super) fn this(&self) -> &JsValue {
        &self.this
    }

    pub(super) fn arguments(&self) -> &[JsValue] {
        match &self.arguments {
            Arguments::Few(values) => values,
            Arguments::Many(values) => values,
        }
    }
}

impl Drop for NativeCall {
    fn drop(&mut self) {
        // SAFETY: this guard exists only while executing a native call, never
        // inside a collection sweep. Moving values out prevents a second drop.
        unsafe {
            std::mem::take(&mut self.this).release_outside_collection();
            std::mem::take(&mut self.function).release_outside_collection();
        }
        match &mut self.arguments {
            Arguments::Few(values) => {
                for value in values.drain(..) {
                    // SAFETY: native calls never execute inside a collection sweep.
                    unsafe { value.release_outside_collection() };
                }
            }
            Arguments::Many(values) => {
                for value in values.drain(..) {
                    // SAFETY: native calls never execute inside a collection sweep.
                    unsafe { value.release_outside_collection() };
                }
            }
        }
    }
}
