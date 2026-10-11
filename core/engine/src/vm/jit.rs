//! Baseline JIT for numeric leaf functions (feature `jit`).
//!
//! An ordinary function counts its calls. At [`THRESHOLD`] its bytecode is translated,
//! in one forward pass, into [`boa_jit::ir`] and handed to [`boa_jit::compile_numeric_leaf`],
//! once. A later call whose arguments are all Numbers, and whose argument count equals
//! the formal parameter count, runs the machine code and pushes its result instead of
//! pushing a frame. Anything else runs in the interpreter.
//!
//! A function that is not a numeric leaf gets baseline code instead: one helper call per
//! instruction and native branches between them, working on the interpreter's own frame.
//! [`function_call`](crate::builtins::function) leaves such a frame at [`ENTRY`], and the run loop
//! enters the code there. The code hands back to the interpreter whenever the frame does
//! something it does not follow, which is always a valid place to resume.

use std::cell::Cell;
use std::ops::ControlFlow;

use boa_jit::NumericLeafFn;
use boa_jit::baseline::{self, BaselineFn, status};
use boa_jit::ir::{BinOp, CmpOp, Insn, MathFn, Op, Rel, Unary, When};

use crate::{
    Context, JsValue,
    builtins::function::OrdinaryFunction,
    error::RuntimeLimitError,
    vm::{
        CodeBlock, CompletionRecord,
        opcode::{Instruction, OPCODE_HANDLERS, Opcode},
    },
};

/// The `pc` of a frame whose baseline code has not started. No bytecode is this long.
pub(crate) const ENTRY: u32 = u32::MAX;

/// Baseline activations allowed on the Rust stack. Deeper calls run in the interpreter.
const DEPTH_LIMIT: u32 = 200;

const HELPERS: baseline::Helpers = baseline::Helpers {
    step: step_helper,
    branch: branch_helper,
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
    /// Baseline code, entered through [`ENTRY`].
    Baseline(BaselineFn),
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
        JitState::Failed | JitState::Baseline(_) => return false,
        JitState::Cold => {
            let calls = code.jit.calls.get().saturating_add(1);
            code.jit.calls.set(calls);
            if calls < THRESHOLD || !code.is_ordinary() {
                return false;
            }
            // Recorded before compiling, so a failure is never retried.
            code.jit.state.set(JitState::Failed);
            boa_jit::set_helpers(HELPERS);
            let Some((insns, uses_math)) = translate(code) else {
                if baseline_enabled()
                    && let Some(compiled) =
                        translate_baseline(code).and_then(|insns| boa_jit::compile_baseline(&insns))
                {
                    code.jit.state.set(JitState::Baseline(compiled));
                }
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

/// Leaves a just-pushed frame at [`ENTRY`] when its code has baseline code.
pub(crate) fn mark_entry(context: &mut Context) {
    let frame = context.vm.frame_mut();
    if matches!(frame.code_block.jit.state.get(), JitState::Baseline(_)) {
        frame.pc = ENTRY;
    }
}

/// Runs the top frame, which is at [`ENTRY`], as baseline code until it hands back.
pub(crate) fn run_frame(context: &mut Context) -> ControlFlow<CompletionRecord> {
    let frame = context.vm.frame_mut();
    frame.pc = 0;
    let JitState::Baseline(compiled) = frame.code_block.jit.state.get() else {
        return ControlFlow::Continue(());
    };
    if context.vm.jit_depth >= DEPTH_LIMIT {
        return ControlFlow::Continue(());
    }
    let Ok(depth) = u32::try_from(context.vm.frames.len()) else {
        return ControlFlow::Continue(());
    };
    context.vm.jit_depth += 1;
    // SAFETY: `compiled` was emitted on this thread for the top frame's code block, whose
    // bytecode it addresses. It only passes `context` back to the helpers below.
    let result = unsafe { compiled(std::ptr::from_mut(context).cast(), depth) };
    context.vm.jit_depth -= 1;
    if result == status::BREAK
        && let Some(record) = context.vm.jit_break.take()
    {
        return ControlFlow::Break(record);
    }
    ControlFlow::Continue(())
}

/// Runs the handler for the instruction at `pc`, as the run loop would.
fn run_instruction(context: &mut Context, pc: u32) -> ControlFlow<CompletionRecord> {
    if context
        .vm
        .interrupt_flag
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
    {
        return context.handle_error(RuntimeLimitError::Interrupted.into());
    }
    let Some(&byte) = context
        .vm
        .frame()
        .code_block
        .bytecode
        .bytes
        .get(pc as usize)
    else {
        return ControlFlow::Continue(());
    };
    OPCODE_HANDLERS[Opcode::decode(byte) as usize](context, pc as usize)
}

/// Where the frame went after an instruction, as a [`status`] value. A frame the
/// instruction pushed is run to completion here first, so the code continues after a call.
fn settle(
    context: &mut Context,
    flow: ControlFlow<CompletionRecord>,
    next: u32,
    target: u32,
    depth: u32,
) -> u32 {
    let mut flow = flow;
    let depth = depth as usize;
    if flow.is_continue() && context.vm.frames.len() > depth {
        if context.vm.jit_depth >= DEPTH_LIMIT {
            return status::EXIT;
        }
        flow = context.run_nested(depth);
    }
    if let ControlFlow::Break(record) = flow {
        context.vm.jit_break = Some(record);
        return status::BREAK;
    }
    if context.vm.frames.len() != depth {
        return status::EXIT;
    }
    let pc = context.vm.frame().pc;
    if pc == next {
        status::NEXT
    } else if pc == target {
        status::TAKEN
    } else {
        status::EXIT
    }
}

unsafe extern "C" fn step_helper(
    context: *mut std::ffi::c_void,
    pc: u32,
    next: u32,
    depth: u32,
) -> u32 {
    // SAFETY: baseline code passes back the context `run_frame` gave it, which is live and
    // not otherwise borrowed while the code runs.
    let context = unsafe { &mut *context.cast::<Context>() };
    let flow = run_instruction(context, pc);
    settle(context, flow, next, ENTRY, depth)
}

unsafe extern "C" fn branch_helper(
    context: *mut std::ffi::c_void,
    pc: u32,
    next: u32,
    target: u32,
    depth: u32,
) -> u32 {
    // SAFETY: as in `step_helper`.
    let context = unsafe { &mut *context.cast::<Context>() };
    let flow = run_instruction(context, pc);
    settle(context, flow, next, target, depth)
}

/// Baseline instructions for `code`, or `None` when it is not an ordinary function.
fn translate_baseline(code: &CodeBlock) -> Option<Vec<baseline::Insn>> {
    if !code.is_ordinary() {
        return None;
    }
    let bytes = &code.bytecode.bytes;
    let mut insns = Vec::new();
    let mut pc = 0;
    while pc < bytes.len() {
        let (instruction, next) = code.bytecode.next_instruction(pc);
        let at = u32::try_from(pc).ok()?;
        let after = u32::try_from(next).ok()?;
        let branch = |target: u32| baseline::Insn::Branch {
            pc: at,
            next: after,
            target,
        };
        insns.push(match instruction {
            Instruction::Jump { address } => baseline::Insn::Jump {
                pc: at,
                target: u32::from(address),
            },
            Instruction::JumpIfTrue { address, .. }
            | Instruction::JumpIfFalse { address, .. }
            | Instruction::JumpIfNotUndefined { address, .. }
            | Instruction::JumpIfNullOrUndefined { address, .. }
            | Instruction::JumpIfNotLessThan { address, .. }
            | Instruction::JumpIfNotLessThanOrEqual { address, .. }
            | Instruction::JumpIfNotGreaterThan { address, .. }
            | Instruction::JumpIfNotGreaterThanOrEqual { address, .. }
            | Instruction::JumpIfNotEqual { address, .. }
            | Instruction::LogicalAnd { address, .. }
            | Instruction::LogicalOr { address, .. }
            | Instruction::Coalesce { address, .. }
            | Instruction::Case { address, .. }
            | Instruction::TemplateLookup { address, .. } => branch(u32::from(address)),
            _ => baseline::Insn::Step {
                pc: at,
                next: after,
            },
        });
        pc = next;
    }
    Some(insns)
}

/// Baseline code is on unless `BOA_JIT_BASELINE=0` is set.
fn baseline_enabled() -> bool {
    thread_local! {
        static ENABLED: bool = std::env::var_os("BOA_JIT_BASELINE").is_none_or(|value| value != "0");
    }
    ENABLED.with(|enabled| *enabled)
}
