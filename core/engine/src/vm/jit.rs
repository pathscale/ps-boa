//! Baseline JIT for numeric leaf functions (feature `jit`).
//!
//! An ordinary function counts its calls. At [`THRESHOLD`] its bytecode is translated,
//! in one forward pass, into [`boa_jit::ir`] and handed to [`boa_jit::compile_numeric_leaf`],
//! once. A later call whose arguments are all Numbers, and whose argument count equals
//! the formal parameter count, runs the machine code and pushes its result instead of
//! pushing a frame. Anything else runs in the interpreter.

use std::cell::Cell;

use boa_jit::NumericLeafFn;
use boa_jit::ir::{BinOp, CmpOp, Insn, MathFn, Op, Rel, Unary, When};

use crate::{
    Context, JsValue,
    builtins::function::OrdinaryFunction,
    vm::{CodeBlock, opcode::Instruction},
};

/// Calls before a function is compiled.
const THRESHOLD: u32 = 1000;

/// Per-[`CodeBlock`] JIT state.
#[derive(Debug, Default)]
pub(crate) struct JitSlot {
    calls: Cell<u32>,
    state: Cell<JitState>,
}

impl Clone for JitSlot {
    /// A copied code block starts cold: its machine code belongs to the original.
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Debug, Default, Clone, Copy)]
enum JitState {
    #[default]
    Cold,
    Failed,
    /// Machine code, and whether it computes `Math.f` inline.
    Compiled(NumericLeafFn, bool),
}

/// Runs `code` as machine code when it is compiled and the call qualifies, leaving the
/// result on the stack in place of the arguments, the function and `this`. Returns
/// `false`, with the stack untouched, when the interpreter must run the call.
pub(crate) fn try_call(
    function: &OrdinaryFunction,
    argument_count: usize,
    context: &mut Context,
) -> bool {
    let code = &*function.code;
    let (compiled, uses_math) = match code.jit.state.get() {
        JitState::Compiled(compiled, uses_math) => (compiled, uses_math),
        JitState::Failed => return false,
        JitState::Cold => {
            let calls = code.jit.calls.get().saturating_add(1);
            code.jit.calls.set(calls);
            if calls < THRESHOLD || !code.is_ordinary() {
                return false;
            }
            // Recorded before compiling, so a failure is never retried.
            code.jit.state.set(JitState::Failed);
            let Some((insns, uses_math)) = translate(code) else {
                return false;
            };
            let Some(compiled) =
                boa_jit::compile_numeric_leaf(insns, code.register_count, code.parameter_length)
            else {
                return false;
            };
            code.jit.state.set(JitState::Compiled(compiled, uses_math));
            (compiled, uses_math)
        }
    };
    // Inline `Math.f` is the intrinsic only while the callee's realm has not touched `Math`
    // and no `with` or sloppy `eval` can shadow the global binding. A numeric leaf runs no
    // user code, so what holds at entry holds for the whole call.
    if uses_math {
        let realm = function.realm();
        let global = realm.environment();
        let static_global = function
            .environments
            .current_declarative_ref(global)
            .is_some_and(|env| !env.with() && !env.poisoned());
        if !realm.math_pristine() || !static_global {
            return false;
        }
    }

    if argument_count != code.parameter_length as usize {
        return false;
    }
    let mut numbers = [0.0f64; 8];
    let mut many = Vec::new();
    let args: &mut [f64] = if argument_count <= numbers.len() {
        &mut numbers[..argument_count]
    } else {
        many.resize(argument_count, 0.0);
        &mut many
    };
    if !context
        .vm
        .stack
        .calling_convention_numbers(argument_count, args)
    {
        return false;
    }
    // SAFETY: `compiled` was produced by `compile_numeric_leaf` on this thread for a function
    // with `parameter_length` parameters, and `args` holds that many f64 values for the call.
    // The machine code only reads them, and the thread's JIT module keeps it alive.
    let result = unsafe { compiled(args.as_ptr()) };
    context
        .vm
        .stack
        .calling_convention_replace_call(argument_count, JsValue::from(result));
    true
}

/// One forward pass over the bytecode, with whether it calls `Math`. `None` at the first
/// opcode outside the subset.
fn translate(code: &CodeBlock) -> Option<(Vec<Insn>, bool)> {
    let bytes = &code.bytecode.bytes;
    let mut insns = Vec::new();
    let mut uses_math = false;
    let mut pc = 0;
    while pc < bytes.len() {
        let (instruction, next) = code.bytecode.next_instruction(pc);
        let op = translate_op(&instruction)?;
        uses_math |= matches!(op, Op::Math { .. });
        insns.push(Insn {
            pc: u32::try_from(pc).ok()?,
            op,
        });
        pc = next;
    }
    Some((insns, uses_math))
}

fn translate_op(instruction: &Instruction) -> Option<Op> {
    let r = |operand| u32::from(operand);
    let bin = |op, dst, lhs, rhs| Op::Bin {
        op,
        dst: r(dst),
        lhs: r(lhs),
        rhs: r(rhs),
    };
    let cmp = |op, dst, lhs, rhs| Op::Cmp {
        op,
        dst: r(dst),
        lhs: r(lhs),
        rhs: r(rhs),
    };
    let rel = |rel, address, lhs, rhs| Op::JumpRel {
        rel,
        target: u32::from(address),
        lhs: r(lhs),
        rhs: r(rhs),
    };
    let constant = |dst, value| Op::Const { dst: r(dst), value };
    Some(match *instruction {
        Instruction::StoreZero { dst } => constant(dst, 0.0),
        Instruction::StoreOne { dst } => constant(dst, 1.0),
        Instruction::StoreInt8 { dst, value } => constant(dst, f64::from(value)),
        Instruction::StoreInt16 { dst, value } => constant(dst, f64::from(value)),
        Instruction::StoreInt32 { dst, value } => constant(dst, f64::from(value)),
        Instruction::StoreFloat { dst, value } => constant(dst, f64::from(value)),
        Instruction::StoreDouble { dst, value } => constant(dst, value),
        Instruction::StoreNan { dst } => constant(dst, f64::NAN),
        Instruction::StorePositiveInfinity { dst } => constant(dst, f64::INFINITY),
        Instruction::StoreNegativeInfinity { dst } => constant(dst, f64::NEG_INFINITY),
        Instruction::Add { dst, lhs, rhs } => bin(BinOp::Add, dst, lhs, rhs),
        Instruction::Sub { dst, lhs, rhs } => bin(BinOp::Sub, dst, lhs, rhs),
        Instruction::Mul { dst, lhs, rhs } => bin(BinOp::Mul, dst, lhs, rhs),
        Instruction::Div { dst, lhs, rhs } => bin(BinOp::Div, dst, lhs, rhs),
        Instruction::Mod { dst, lhs, rhs } => bin(BinOp::Mod, dst, lhs, rhs),
        Instruction::BitAnd { dst, lhs, rhs } => bin(BinOp::BitAnd, dst, lhs, rhs),
        Instruction::BitOr { dst, lhs, rhs } => bin(BinOp::BitOr, dst, lhs, rhs),
        Instruction::BitXor { dst, lhs, rhs } => bin(BinOp::BitXor, dst, lhs, rhs),
        Instruction::ShiftLeft { dst, lhs, rhs } => bin(BinOp::Shl, dst, lhs, rhs),
        Instruction::ShiftRight { dst, lhs, rhs } => bin(BinOp::Shr, dst, lhs, rhs),
        Instruction::UnsignedShiftRight { dst, lhs, rhs } => bin(BinOp::Ushr, dst, lhs, rhs),
        Instruction::Eq { dst, lhs, rhs } => cmp(CmpOp::Eq, dst, lhs, rhs),
        Instruction::NotEq { dst, lhs, rhs } => cmp(CmpOp::Ne, dst, lhs, rhs),
        Instruction::StrictEq { dst, lhs, rhs } => cmp(CmpOp::StrictEq, dst, lhs, rhs),
        Instruction::StrictNotEq { dst, lhs, rhs } => cmp(CmpOp::StrictNe, dst, lhs, rhs),
        Instruction::LessThan { dst, lhs, rhs } => cmp(CmpOp::Lt, dst, lhs, rhs),
        Instruction::LessThanOrEq { dst, lhs, rhs } => cmp(CmpOp::Le, dst, lhs, rhs),
        Instruction::GreaterThan { dst, lhs, rhs } => cmp(CmpOp::Gt, dst, lhs, rhs),
        Instruction::GreaterThanOrEq { dst, lhs, rhs } => cmp(CmpOp::Ge, dst, lhs, rhs),
        Instruction::Pos { value } => Op::Unary {
            op: Unary::Pos,
            reg: r(value),
        },
        Instruction::Neg { value } => Op::Unary {
            op: Unary::Neg,
            reg: r(value),
        },
        Instruction::BitNot { value } => Op::BitNot { reg: r(value) },
        Instruction::Inc { dst, src } => Op::Inc {
            dst: r(dst),
            src: r(src),
            dec: false,
        },
        Instruction::Dec { dst, src } => Op::Inc {
            dst: r(dst),
            src: r(src),
            dec: true,
        },
        Instruction::GetArgument { index, dst } => Op::GetArg {
            index: u32::from(index),
            dst: r(dst),
        },
        Instruction::Move { dst, src } => Op::Move {
            dst: r(dst),
            src: r(src),
        },
        Instruction::Jump { address } => Op::Jump {
            target: u32::from(address),
        },
        Instruction::JumpIfTrue { address, value } => Op::JumpIf {
            when: When::True,
            target: u32::from(address),
            value: r(value),
        },
        Instruction::JumpIfFalse { address, value } => Op::JumpIf {
            when: When::False,
            target: u32::from(address),
            value: r(value),
        },
        Instruction::JumpIfNotLessThan { address, lhs, rhs } => rel(Rel::Lt, address, lhs, rhs),
        Instruction::JumpIfNotLessThanOrEqual { address, lhs, rhs } => {
            rel(Rel::Le, address, lhs, rhs)
        }
        Instruction::JumpIfNotGreaterThan { address, lhs, rhs } => rel(Rel::Gt, address, lhs, rhs),
        Instruction::JumpIfNotGreaterThanOrEqual { address, lhs, rhs } => {
            rel(Rel::Ge, address, lhs, rhs)
        }
        Instruction::JumpIfNotEqual { address, lhs, rhs } => Op::JumpNe {
            target: u32::from(address),
            lhs: r(lhs),
            rhs: r(rhs),
        },
        Instruction::SetAccumulator { src } => Op::SetAcc { src: r(src) },
        Instruction::CheckReturn => Op::CheckReturn,
        Instruction::Return => Op::Return,
        Instruction::IncrementLoopIteration => Op::LoopIter,
        // The callee and `this` registers PrepareMath fills are read only by CallMath.
        Instruction::PrepareMath { .. } => Op::Nop,
        Instruction::CallMath { operands } => {
            let name = crate::builtins::math::Math::pure_name(operands.function);
            let function = match name.to_std_string_escaped().as_str() {
                "max" => MathFn::Max,
                "min" => MathFn::Min,
                "floor" => MathFn::Floor,
                "ceil" => MathFn::Ceil,
                "abs" => MathFn::Abs,
                "sqrt" => MathFn::Sqrt,
                "trunc" => MathFn::Trunc,
                _ => return None,
            };
            Op::Math {
                f: function,
                dst: r(operands.dst),
                args: operands.args.map(r),
                argc: operands.argc,
            }
        }
        _ => return None,
    })
}
