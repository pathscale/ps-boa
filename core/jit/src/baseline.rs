//! Baseline code: every bytecode instruction becomes a direct call to an engine helper
//! that runs the interpreter's handler for it, with control flow between instructions
//! compiled to native branches.
//!
//! The code works on the interpreter's own frame and registers, so returning to the
//! interpreter at any instruction boundary is always valid. Each helper reports what the
//! code does next with a [`Status`] value.

use cranelift_codegen::ir::{self, InstBuilder, types};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{FuncId, Module};
use std::collections::HashMap;

/// What a helper tells the compiled code to do next.
pub mod status {
    /// Continue with the next instruction.
    pub const NEXT: u32 = 0;
    /// A branch was taken: continue at its target.
    pub const TAKEN: u32 = 1;
    /// Leave the compiled code; the interpreter resumes from the current frame state.
    pub const EXIT: u32 = 2;
    /// Leave the compiled code; the engine holds a completion its run loop must return.
    pub const BREAK: u32 = 3;
}

/// Where the running function's registers are. The engine rewrites it before the code
/// starts and before every helper returns [`status::NEXT`] or [`status::TAKEN`], since a
/// helper can move the stack; the code reloads it after each helper call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct JitFrame {
    /// Register 0 of the running frame. Each register is one boxed value (`u64`).
    pub regs: *mut u64,
    /// The call's argument count. Argument `i` is at `regs[i - argc]` when `i < argc`.
    pub argc: u32,
}

/// Machine code for one function body.
///
/// `context` is the engine's context, passed through to the helpers untouched. `depth`
/// is the frame count while the function's own frame is on top. `frame` points at the
/// engine's [`JitFrame`]. Returns a [`status`] value, never [`status::NEXT`] or
/// [`status::TAKEN`].
pub type BaselineFn =
    unsafe extern "C" fn(context: *mut std::ffi::c_void, depth: u32, frame: *const JitFrame) -> u32;

/// The boxed value encoding the inline paths read and write: a NaN-boxed `u64`.
pub mod boxed {
    /// Exponent bits of a `NaN`.
    pub const MASK_NAN: u64 = 0x7FF0_0000_0000_0000;
    /// Exponent bits plus the 4-bit tag.
    pub const MASK_KIND: u64 = MASK_NAN | 0xF_0000_0000_0000;
    /// `value & MASK_KIND == KIND_INT32`: an int32 in the low 32 bits.
    pub const KIND_INT32: u64 = MASK_NAN | 0x9_0000_0000_0000;
    /// `value & MASK_KIND == KIND_BOOLEAN`: a boolean in bit 0.
    pub const KIND_BOOLEAN: u64 = MASK_NAN | 0xA_0000_0000_0000;
    /// `value & MASK_POINTER == MASK_POINTER`: object, string, symbol or `BigInt`. Such a
    /// value owns a reference count; the inline paths never copy or overwrite one.
    pub const MASK_POINTER: u64 = MASK_NAN | 0xC_0000_0000_0000;
    /// The one `NaN` a float64 may hold. Every other `NaN` bit pattern must become this.
    pub const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;
    /// `undefined`.
    pub const UNDEFINED: u64 = MASK_NAN | 0xB_0000_0000_0001;
    /// A value is a float64 when `value & MASK_NAN != MASK_NAN`, or its kind is
    /// `MASK_NAN` (infinities) or `CANONICAL_NAN`.
    pub const KIND_INFINITY: u64 = MASK_NAN;
}

/// `step(context, pc, next, depth)`: run the instruction at `pc`. Returns
/// [`status::NEXT`] when the frame continues at `next`, else `EXIT` or `BREAK`.
pub type StepFn = unsafe extern "C" fn(*mut std::ffi::c_void, u32, u32, u32) -> u32;

/// `branch(context, pc, next, target, depth)`: run the branching instruction at `pc`.
/// Returns [`status::NEXT`] or [`status::TAKEN`] for the two successors, else `EXIT` or
/// `BREAK`.
pub type BranchFn = unsafe extern "C" fn(*mut std::ffi::c_void, u32, u32, u32, u32) -> u32;

/// The engine helpers baseline code calls.
#[derive(Debug, Clone, Copy)]
pub struct Helpers {
    /// Runs one instruction.
    pub step: StepFn,
    /// Runs one branching instruction.
    pub branch: BranchFn,
}

/// A two-operand Number operation with JavaScript semantics.
#[derive(Debug, Clone, Copy)]
pub enum Arith {
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
}

/// The relation a fused compare-and-jump tests. The jump is taken when it does NOT hold.
#[derive(Debug, Clone, Copy)]
pub enum Rel {
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
}

/// An instruction with an inline fast path. When the fast path does not apply, the code
/// runs the instruction through the step helper instead, as an [`Insn::Step`].
#[derive(Debug, Clone, Copy)]
pub enum FastOp {
    /// `regs[dst] = regs[src]`.
    Move {
        /// Destination register.
        dst: u32,
        /// Source register.
        src: u32,
    },
    /// `regs[dst] = bits`, a boxed number.
    Const {
        /// Destination register.
        dst: u32,
        /// The boxed value, never a pointer kind.
        bits: u64,
    },
    /// `regs[dst] = regs[lhs] op regs[rhs]`.
    Arith {
        /// The operation.
        op: Arith,
        /// Destination register.
        dst: u32,
        /// Left operand register.
        lhs: u32,
        /// Right operand register.
        rhs: u32,
    },
    /// `regs[dst] = regs[src] + 1`, or `- 1` when `dec`.
    Inc {
        /// Destination register.
        dst: u32,
        /// Source register.
        src: u32,
        /// Decrement instead.
        dec: bool,
    },
    /// `regs[dst] = argument index`, or `undefined` past the argument count.
    GetArg {
        /// Formal parameter index.
        index: u32,
        /// Destination register.
        dst: u32,
    },
}

/// A branching instruction with an inline fast path. When the fast path does not apply,
/// the code runs it through the branch helper instead, as an [`Insn::Branch`].
#[derive(Debug, Clone, Copy)]
pub enum FastCond {
    /// Jump unless `regs[lhs] rel regs[rhs]`.
    JumpUnless {
        /// The relation.
        rel: Rel,
        /// Left operand register.
        lhs: u32,
        /// Right operand register.
        rhs: u32,
    },
    /// Jump when `regs[value]` is truthy (`when_true`) or falsy.
    JumpIf {
        /// Jump on truthy rather than falsy.
        when_true: bool,
        /// Tested register.
        value: u32,
    },
}

/// One instruction of a function body.
#[derive(Debug, Clone, Copy)]
pub enum Insn {
    /// Run the instruction at `pc` through the step helper.
    Step {
        /// Bytecode address.
        pc: u32,
        /// Address of the next instruction.
        next: u32,
    },
    /// Run the branching instruction at `pc` through the branch helper.
    Branch {
        /// Bytecode address.
        pc: u32,
        /// Address of the next instruction.
        next: u32,
        /// Address the branch goes to when taken.
        target: u32,
    },
    /// Unconditional jump with no other effect.
    Jump {
        /// Bytecode address.
        pc: u32,
        /// Target address.
        target: u32,
    },
    /// An instruction with an inline fast path.
    Fast {
        /// Bytecode address.
        pc: u32,
        /// Address of the next instruction.
        next: u32,
        /// The operation.
        op: FastOp,
    },
    /// A branching instruction with an inline fast path.
    FastBranch {
        /// Bytecode address.
        pc: u32,
        /// Address of the next instruction.
        next: u32,
        /// Address the branch goes to when taken.
        target: u32,
        /// The condition.
        cond: FastCond,
    },
}

impl Insn {
    fn pc(self) -> u32 {
        match self {
            Self::Step { pc, .. }
            | Self::Branch { pc, .. }
            | Self::Jump { pc, .. }
            | Self::Fast { pc, .. }
            | Self::FastBranch { pc, .. } => pc,
        }
    }
}

/// `true` when every jump target names an instruction and every register is below
/// `register_count`.
pub(crate) fn validate(insns: &[Insn], register_count: u32) -> bool {
    if insns.is_empty() {
        return false;
    }
    let starts: std::collections::HashSet<u32> = insns.iter().map(|insn| insn.pc()).collect();
    let reg = |r: u32| r < register_count;
    insns.iter().all(|insn| match *insn {
        Insn::Step { .. } => true,
        Insn::Fast { op, .. } => match op {
            FastOp::Move { dst, src } | FastOp::Inc { dst, src, .. } => reg(dst) && reg(src),
            FastOp::Const { dst, bits } => {
                reg(dst) && bits & boxed::MASK_POINTER != boxed::MASK_POINTER
            }
            FastOp::Arith { dst, lhs, rhs, .. } => reg(dst) && reg(lhs) && reg(rhs),
            FastOp::GetArg { dst, .. } => reg(dst),
        },
        Insn::FastBranch { target, cond, .. } => {
            starts.contains(&target)
                && match cond {
                    FastCond::JumpUnless { lhs, rhs, .. } => reg(lhs) && reg(rhs),
                    FastCond::JumpIf { value, .. } => reg(value),
                }
        }
        Insn::Branch { target, .. } | Insn::Jump { target, .. } => starts.contains(&target),
    })
}

/// Emits `fn(context, depth, frame) -> status` for `insns`.
pub(crate) fn emit(
    builder: &mut FunctionBuilder<'_>,
    module: &mut impl Module,
    step: FuncId,
    branch: FuncId,
    insns: &[Insn],
) {
    let step = module.declare_func_in_func(step, builder.func);
    let branch = module.declare_func_in_func(branch, builder.func);

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let context = builder.block_params(entry)[0];
    let depth = builder.block_params(entry)[1];

    let blocks: Vec<ir::Block> = insns.iter().map(|_| builder.create_block()).collect();
    let index: HashMap<u32, usize> = insns
        .iter()
        .enumerate()
        .map(|(i, insn)| (insn.pc(), i))
        .collect();
    // Falling off the end leaves the frame to the interpreter.
    let exit = builder.create_block();
    builder.append_block_param(exit, types::I32);
    let block_at = |pc: u32| index.get(&pc).map(|&i| blocks[i]);

    builder.ins().jump(blocks[0], &[]);

    for (i, insn) in insns.iter().enumerate() {
        builder.switch_to_block(blocks[i]);
        let fall = blocks.get(i + 1).copied();
        let exit_code = |builder: &mut FunctionBuilder<'_>| {
            builder.ins().iconst(types::I32, i64::from(status::EXIT))
        };
        // Inline fast paths are not emitted yet: they run through the helpers.
        let insn = match *insn {
            Insn::Fast { pc, next, .. } => Insn::Step { pc, next },
            Insn::FastBranch {
                pc, next, target, ..
            } => Insn::Branch { pc, next, target },
            other => other,
        };
        match insn {
            Insn::Step { pc, next } => {
                let pc = builder.ins().iconst(types::I32, i64::from(pc));
                let next_value = builder.ins().iconst(types::I32, i64::from(next));
                let call = builder.ins().call(step, &[context, pc, next_value, depth]);
                let result = builder.inst_results(call)[0];
                if let Some(target) = block_at(next).or(fall) {
                    builder
                        .ins()
                        .brif(result, exit, &[result.into()], target, &[]);
                } else {
                    let code = exit_code(builder);
                    let chosen = builder.ins().select(result, result, code);
                    builder.ins().jump(exit, &[chosen.into()]);
                }
            }
            Insn::Branch { pc, next, target } => {
                let pc = builder.ins().iconst(types::I32, i64::from(pc));
                let next_value = builder.ins().iconst(types::I32, i64::from(next));
                let target_value = builder.ins().iconst(types::I32, i64::from(target));
                let call = builder
                    .ins()
                    .call(branch, &[context, pc, next_value, target_value, depth]);
                let result = builder.inst_results(call)[0];
                let next_block = block_at(next).unwrap_or(exit);
                let not_next = builder.create_block();
                let code = exit_code(builder);
                let next_args: Vec<ir::BlockArg> = if next_block == exit {
                    vec![code.into()]
                } else {
                    Vec::new()
                };
                builder
                    .ins()
                    .brif(result, not_next, &[], next_block, &next_args);
                builder.switch_to_block(not_next);
                let taken = builder.ins().icmp_imm_u(
                    ir::condcodes::IntCC::Equal,
                    result,
                    i64::from(status::TAKEN),
                );
                let leave = builder.create_block();
                let taken_block = block_at(target).unwrap_or(leave);
                builder.ins().brif(taken, taken_block, &[], leave, &[]);
                builder.switch_to_block(leave);
                builder.ins().jump(exit, &[result.into()]);
            }
            Insn::Fast { .. } | Insn::FastBranch { .. } => unreachable!(),
            Insn::Jump { target, .. } => {
                if let Some(target) = block_at(target) {
                    builder.ins().jump(target, &[]);
                } else {
                    let code = exit_code(builder);
                    builder.ins().jump(exit, &[code.into()]);
                }
            }
        }
    }

    builder.switch_to_block(exit);
    let result = builder.block_params(exit)[0];
    builder.ins().return_(&[result]);
}
