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

mod typed;

/// The `pc` of a frame whose baseline code has not started. No bytecode is this long.
pub(crate) const ENTRY: u32 = u32::MAX;

/// Baseline activations allowed on the Rust stack. Deeper calls run in the interpreter.
const DEPTH_LIMIT: u32 = 200;

const HELPERS: baseline::Helpers = baseline::Helpers {
    step: step_helper,
    branch: branch_helper,
    calls: typed::HELPERS,
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
                    && let Some(compiled) = translate_baseline(code)
                        .and_then(|insns| boa_jit::compile_baseline(&insns, code.register_count))
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
    refresh_frame(context);
    let frame = std::ptr::from_ref(&context.vm.jit_frame);
    // SAFETY: `compiled` was emitted on this thread for the top frame's code block, whose
    // bytecode it addresses. It only passes `context` back to the helpers below.
    let result = unsafe { compiled(std::ptr::from_mut(context).cast(), depth, frame) };
    context.vm.jit_depth -= 1;
    if result == status::BREAK {
        return take_break(context);
    }
    ControlFlow::Continue(())
}

#[cold]
fn take_break(context: &mut Context) -> ControlFlow<CompletionRecord> {
    match context.vm.jit_break.take() {
        Some(record) => ControlFlow::Break(record),
        None => ControlFlow::Continue(()),
    }
}

/// Points [`Vm::jit_frame`](crate::vm::Vm) at the top frame's registers.
fn refresh_frame(context: &mut Context) {
    let frame = context.vm.frame();
    let (rp, argc) = (frame.rp as usize, frame.argument_count);
    let stack = context.vm.stack.stack.as_mut_ptr();
    context.vm.jit_frame = baseline::JitFrame {
        regs: stack.wrapping_add(rp).cast::<u64>(),
        argc,
        stack: stack.cast::<u64>(),
        depth: context.vm.frames.len(),
    };
}

#[inline]
fn interrupt_pending(context: &Context) -> bool {
    context
        .vm
        .interrupt_flag
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        || context.vm.execution_budget.should_interrupt()
}

#[cold]
fn interrupted(context: &mut Context, next: u32, target: u32, depth: u32) -> u32 {
    let flow = context.handle_error(RuntimeLimitError::Interrupted.into());
    if flow.is_break() {
        return save_break(context, flow);
    }
    settle(context, next, target, depth)
}

/// Only the cold path transfers ownership of a completion record.
#[cold]
fn save_break(context: &mut Context, flow: ControlFlow<CompletionRecord>) -> u32 {
    let ControlFlow::Break(record) = flow else {
        unreachable!("only a break completion is saved");
    };
    context.vm.jit_break = Some(record);
    status::BREAK
}

/// Runs the handler for the instruction at `pc`, as the run loop would.
#[inline]
fn run_step(context: &mut Context, pc: u32, next: u32, target: u32, depth: u32) -> u32 {
    if interrupt_pending(context) {
        return interrupted(context, next, target, depth);
    }
    let Some(&byte) = context
        .vm
        .frame()
        .code_block
        .bytecode
        .bytes
        .get(pc as usize)
    else {
        return settle(context, next, target, depth);
    };
    let flow = OPCODE_HANDLERS[Opcode::decode(byte) as usize](context, pc as usize);
    if flow.is_break() {
        return save_break(context, flow);
    }
    settle(context, next, target, depth)
}

/// Where the frame went after a continuing instruction, as a [`status`] value.
#[inline]
fn settle(context: &mut Context, next: u32, target: u32, depth: u32) -> u32 {
    let count = context.vm.frames.len();
    if count != depth as usize {
        return settle_frames(context, next, target, depth);
    }

    // Read the top frame once. Nested baseline activations overwrite the shared
    // register view, so its cached depth must be checked even if the stack did not move.
    let frame = context.vm.frames.last().expect("a baseline frame is live");
    let pc = frame.pc;
    let stack = context.vm.stack.stack.as_mut_ptr();
    if context.vm.jit_frame.stack != stack.cast::<u64>() || context.vm.jit_frame.depth != count {
        context.vm.jit_frame = baseline::JitFrame {
            regs: stack.wrapping_add(frame.rp as usize).cast::<u64>(),
            argc: frame.argument_count,
            stack: stack.cast::<u64>(),
            depth: count,
        };
    }
    if pc == next {
        status::NEXT
    } else if pc == target {
        status::TAKEN
    } else {
        status::EXIT
    }
}

/// A frame the instruction pushed is run to completion before resuming its caller.
#[cold]
fn settle_frames(context: &mut Context, next: u32, target: u32, depth: u32) -> u32 {
    if context.vm.frames.len() > depth as usize {
        if context.vm.jit_depth >= DEPTH_LIMIT {
            return status::EXIT;
        }
        let flow = context.run_nested(depth as usize);
        if flow.is_break() {
            return save_break(context, flow);
        }
    }
    if context.vm.frames.len() != depth as usize {
        return status::EXIT;
    }
    settle(context, next, target, depth)
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
    run_step(context, pc, next, ENTRY, depth)
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
    run_step(context, pc, next, target, depth)
}

/// Baseline instructions for `code`, or `None` when it is not an ordinary function.
fn translate_baseline(code: &CodeBlock) -> Option<Vec<baseline::Insn>> {
    if !code.is_ordinary() {
        return None;
    }
    let bytes = &code.bytecode.bytes;
    // A rest parameter moves the arguments, so `GetArgument` runs through its handler.
    let mut rest = false;
    let mut pc = 0;
    while pc < bytes.len() {
        let (instruction, next) = code.bytecode.next_instruction(pc);
        rest |= matches!(instruction, Instruction::RestParameterInit { .. });
        pc = next;
    }
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
        if let Some(insn) = fast(&instruction, at, after, rest)
            .or_else(|| typed::translate(&instruction, at, after))
        {
            insns.push(insn);
            pc = next;
            continue;
        }
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

/// The inline form of `instruction`, when it has one. Inline paths read NaN-boxed
/// registers, so the enum value representation has none.
#[cfg(all(not(feature = "jsvalue-enum"), target_pointer_width = "64"))]
fn fast(instruction: &Instruction, pc: u32, next: u32, rest: bool) -> Option<baseline::Insn> {
    use baseline::{Arith, FastCond, FastOp, Insn, Rel};
    let r = |operand| u32::from(operand);
    let constant = |dst, value: JsValue| FastOp::Const {
        dst: r(dst),
        bits: value.jit_bits(),
    };
    let arith = |op, dst, lhs, rhs| FastOp::Arith {
        op,
        dst: r(dst),
        lhs: r(lhs),
        rhs: r(rhs),
    };
    let unless = |rel, address, lhs, rhs| Insn::FastBranch {
        pc,
        next,
        target: u32::from(address),
        cond: FastCond::JumpUnless {
            rel,
            lhs: r(lhs),
            rhs: r(rhs),
        },
    };
    let jump_if = |when_true, address, value| Insn::FastBranch {
        pc,
        next,
        target: u32::from(address),
        cond: FastCond::JumpIf {
            when_true,
            value: r(value),
        },
    };
    let op = match *instruction {
        Instruction::Move { dst, src } => FastOp::Move {
            dst: r(dst),
            src: r(src),
        },
        Instruction::StoreZero { dst } => constant(dst, JsValue::from(0)),
        Instruction::StoreOne { dst } => constant(dst, JsValue::from(1)),
        Instruction::StoreInt8 { dst, value } => constant(dst, JsValue::from(i32::from(value))),
        Instruction::StoreInt16 { dst, value } => constant(dst, JsValue::from(i32::from(value))),
        Instruction::StoreInt32 { dst, value } => constant(dst, JsValue::from(value)),
        Instruction::StoreFloat { dst, value } => constant(dst, JsValue::from(value)),
        Instruction::StoreDouble { dst, value } => constant(dst, JsValue::from(value)),
        Instruction::StoreNan { dst } => constant(dst, JsValue::nan()),
        Instruction::StorePositiveInfinity { dst } => constant(dst, JsValue::positive_infinity()),
        Instruction::StoreNegativeInfinity { dst } => constant(dst, JsValue::negative_infinity()),
        Instruction::Add { dst, lhs, rhs } => arith(Arith::Add, dst, lhs, rhs),
        Instruction::Sub { dst, lhs, rhs } => arith(Arith::Sub, dst, lhs, rhs),
        Instruction::Mul { dst, lhs, rhs } => arith(Arith::Mul, dst, lhs, rhs),
        Instruction::Div { dst, lhs, rhs } => arith(Arith::Div, dst, lhs, rhs),
        Instruction::Inc { dst, src } => FastOp::Inc {
            dst: r(dst),
            src: r(src),
            dec: false,
        },
        Instruction::Dec { dst, src } => FastOp::Inc {
            dst: r(dst),
            src: r(src),
            dec: true,
        },
        Instruction::GetArgument { index, dst } if !rest => FastOp::GetArg {
            index: u32::from(index),
            dst: r(dst),
        },
        Instruction::JumpIfNotLessThan { address, lhs, rhs } => {
            return Some(unless(Rel::Lt, address, lhs, rhs));
        }
        Instruction::JumpIfNotLessThanOrEqual { address, lhs, rhs } => {
            return Some(unless(Rel::Le, address, lhs, rhs));
        }
        Instruction::JumpIfNotGreaterThan { address, lhs, rhs } => {
            return Some(unless(Rel::Gt, address, lhs, rhs));
        }
        Instruction::JumpIfNotGreaterThanOrEqual { address, lhs, rhs } => {
            return Some(unless(Rel::Ge, address, lhs, rhs));
        }
        Instruction::JumpIfTrue { address, value } => return Some(jump_if(true, address, value)),
        Instruction::JumpIfFalse { address, value } => return Some(jump_if(false, address, value)),
        _ => return None,
    };
    Some(Insn::Fast { pc, next, op })
}

#[cfg(not(all(not(feature = "jsvalue-enum"), target_pointer_width = "64")))]
fn fast(_: &Instruction, _: u32, _: u32, _: bool) -> Option<baseline::Insn> {
    None
}

/// Baseline code is on unless `BOA_JIT_BASELINE=0` is set.
fn baseline_enabled() -> bool {
    thread_local! {
        static ENABLED: bool = std::env::var_os("BOA_JIT_BASELINE").is_none_or(|value| value != "0");
    }
    ENABLED.with(|enabled| *enabled)
}
