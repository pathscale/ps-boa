//! Baseline helpers with operands decoded during translation.

use boa_jit::baseline::{CallFn, HelperKind, Insn};

use crate::{
    Context,
    vm::{
        completion_record::IntoCompletionRecord,
        opcode::{
            CheckReturn, GetName, GetNameGlobal, GetPropertyByIndex, GetPropertyByName,
            GetPropertyByValue, Instruction, PopIntoRegister, PushFromRegister, PushValueToArray,
            SetAccumulator,
        },
    },
};

use super::{ENTRY, interrupt_pending, interrupted, save_break, settle};

// The order matches HelperKind::ALL.
pub(super) const HELPERS: [CallFn; HelperKind::COUNT] = [
    get_property_by_index,
    get_property_by_name,
    get_name,
    get_name_global,
    push_from_register,
    pop_into_register,
    set_accumulator,
    check_return,
    get_property_by_value,
    push_value_to_array,
];

macro_rules! helper {
    ($name:ident($a:ident, $b:ident, $c:ident, $d:ident), $op:ident, $args:expr) => {
        unsafe extern "C" fn $name(
            context: *mut std::ffi::c_void,
            next: u32,
            depth: u32,
            $a: u32,
            $b: u32,
            $c: u32,
            $d: u32,
        ) -> u32 {
            // SAFETY: baseline code passes the live context supplied by run_frame.
            let context = unsafe { &mut *context.cast::<Context>() };
            if interrupt_pending(context) {
                return interrupted(context, next, ENTRY, depth);
            }
            // Match the generated opcode handler, including the PC used for errors.
            context.vm.frame_mut().pc = next;
            let result = $op::operation($args, context);
            let flow = result.into_completion_record(context);
            if flow.is_break() {
                return save_break(context, flow);
            }
            settle(context, next, ENTRY, depth)
        }
    };
}

helper!(
    get_property_by_index(a, b, c, _d),
    GetPropertyByIndex,
    (a.into(), b.into(), c.into())
);
helper!(
    get_property_by_name(a, b, c, _d),
    GetPropertyByName,
    (a.into(), b.into(), c.into())
);
helper!(get_name(a, b, _c, _d), GetName, (a.into(), b.into()));
helper!(
    get_name_global(a, b, c, _d),
    GetNameGlobal,
    (a.into(), b.into(), c.into())
);
helper!(
    push_from_register(a, _b, _c, _d),
    PushFromRegister,
    a.into()
);
helper!(pop_into_register(a, _b, _c, _d), PopIntoRegister, a.into());
helper!(set_accumulator(a, _b, _c, _d), SetAccumulator, a.into());
helper!(check_return(_a, _b, _c, _d), CheckReturn, ());
helper!(
    get_property_by_value(a, b, c, d),
    GetPropertyByValue,
    (a.into(), b.into(), c.into(), d.into())
);
helper!(
    push_value_to_array(a, b, _c, _d),
    PushValueToArray,
    (a.into(), b.into())
);

pub(super) fn translate(instruction: &Instruction, pc: u32, next: u32) -> Option<Insn> {
    let (helper, args) = match *instruction {
        Instruction::GetPropertyByIndex { dst, object, index } => (
            HelperKind::GetPropertyByIndex,
            [u32::from(dst), u32::from(object), u32::from(index), 0],
        ),
        Instruction::GetPropertyByName {
            dst,
            value,
            ic_index,
        } => (
            HelperKind::GetPropertyByName,
            [u32::from(dst), u32::from(value), u32::from(ic_index), 0],
        ),
        Instruction::GetName { dst, binding_index } => (
            HelperKind::GetName,
            [u32::from(dst), u32::from(binding_index), 0, 0],
        ),
        Instruction::GetNameGlobal {
            dst,
            binding_index,
            ic_index,
        } => (
            HelperKind::GetNameGlobal,
            [
                u32::from(dst),
                u32::from(binding_index),
                u32::from(ic_index),
                0,
            ],
        ),
        Instruction::PushFromRegister { src } => {
            (HelperKind::PushFromRegister, [u32::from(src), 0, 0, 0])
        }
        Instruction::PopIntoRegister { dst } => {
            (HelperKind::PopIntoRegister, [u32::from(dst), 0, 0, 0])
        }
        Instruction::SetAccumulator { src } => {
            (HelperKind::SetAccumulator, [u32::from(src), 0, 0, 0])
        }
        Instruction::CheckReturn => (HelperKind::CheckReturn, [0; 4]),
        Instruction::GetPropertyByValue {
            dst,
            key,
            receiver,
            object,
        } => (
            HelperKind::GetPropertyByValue,
            [
                u32::from(dst),
                u32::from(key),
                u32::from(receiver),
                u32::from(object),
            ],
        ),
        Instruction::PushValueToArray { value, array } => (
            HelperKind::PushValueToArray,
            [u32::from(value), u32::from(array), 0, 0],
        ),
        _ => return None,
    };
    Some(Insn::Call1 {
        pc,
        next,
        helper,
        args,
    })
}
