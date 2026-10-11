//! Baseline code: numeric operations and branches have inline fast paths, while other
//! bytecode instructions call engine helpers that run the interpreter's handlers.
//! Control flow between instructions is compiled to native branches.
//!
//! The code works on the interpreter's own frame and registers, so returning to the
//! interpreter at any instruction boundary is always valid. Each helper reports what the
//! code does next with a [`status`] value.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::Ieee64;
use cranelift_codegen::ir::{self, InstBuilder, MemFlagsData, types};
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

/// Where the running function's registers are. The engine initializes it on entry and
/// refreshes it after a helper when the stack buffer or cached frame count changed.
/// The code reloads the register view after each helper call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct JitFrame {
    /// Register 0 of the running frame. Each register is one boxed value (`u64`).
    pub regs: *mut u64,
    /// The call's argument count. Argument `i` is at `regs[i - argc]` when `i < argc`.
    pub argc: u32,
    /// Stack buffer used to form `regs`, compared by the engine without dereferencing it.
    pub stack: *mut u64,
    /// Frame count when this view was refreshed, including nested baseline entries.
    pub depth: usize,
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

/// `call(context, next, depth, a, b, c, d)`: run an operation with decoded operands.
/// The helper advances the frame to `next` before executing the operation. Unused
/// operands are zero. Returns [`status::NEXT`], `EXIT` or `BREAK`.
pub type CallFn = unsafe extern "C" fn(*mut std::ffi::c_void, u32, u32, u32, u32, u32, u32) -> u32;

/// An operation whose helper accepts decoded operands.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum HelperKind {
    /// Operands: destination register, object register, constant index.
    GetPropertyByIndex,
    /// Operands: destination register, value register, inline cache index.
    GetPropertyByName,
    /// Operands: destination register, binding index.
    GetName,
    /// Operands: destination register, binding index, inline cache index.
    GetNameGlobal,
    /// Operand: source register.
    PushFromRegister,
    /// Operand: destination register.
    PopIntoRegister,
    /// Operand: source register.
    SetAccumulator,
    /// No operands.
    CheckReturn,
    /// Operands: destination, key, receiver and object registers.
    GetPropertyByValue,
    /// Operands: value register, array register.
    PushValueToArray,
}

impl HelperKind {
    /// Number of decoded-operand helpers.
    pub const COUNT: usize = 10;

    /// Helpers in registration order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::GetPropertyByIndex,
        Self::GetPropertyByName,
        Self::GetName,
        Self::GetNameGlobal,
        Self::PushFromRegister,
        Self::PopIntoRegister,
        Self::SetAccumulator,
        Self::CheckReturn,
        Self::GetPropertyByValue,
        Self::PushValueToArray,
    ];

    /// Index in the helper registration array.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Imported symbol used by generated code.
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::GetPropertyByIndex => "boa_jit_get_property_by_index",
            Self::GetPropertyByName => "boa_jit_get_property_by_name",
            Self::GetName => "boa_jit_get_name",
            Self::GetNameGlobal => "boa_jit_get_name_global",
            Self::PushFromRegister => "boa_jit_push_from_register",
            Self::PopIntoRegister => "boa_jit_pop_into_register",
            Self::SetAccumulator => "boa_jit_set_accumulator",
            Self::CheckReturn => "boa_jit_check_return",
            Self::GetPropertyByValue => "boa_jit_get_property_by_value",
            Self::PushValueToArray => "boa_jit_push_value_to_array",
        }
    }

    // Register operands precede all index operands in each helper's argument list.
    const fn register_count(self) -> usize {
        match self {
            Self::GetPropertyByIndex | Self::GetPropertyByName | Self::PushValueToArray => 2,
            Self::GetName
            | Self::GetNameGlobal
            | Self::PushFromRegister
            | Self::PopIntoRegister
            | Self::SetAccumulator => 1,
            Self::CheckReturn => 0,
            Self::GetPropertyByValue => 4,
        }
    }
}

/// The engine helpers baseline code calls.
#[derive(Debug, Clone, Copy)]
pub struct Helpers {
    /// Runs one instruction.
    pub step: StepFn,
    /// Runs one branching instruction.
    pub branch: BranchFn,
    /// Runs decoded operations, indexed by [`HelperKind::index`].
    pub calls: [CallFn; HelperKind::COUNT],
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
    /// Run an operation with decoded operands through its imported helper.
    Call1 {
        /// Bytecode address.
        pc: u32,
        /// Address of the next instruction.
        next: u32,
        /// The operation's helper symbol.
        helper: HelperKind,
        /// Decoded operands in opcode order, padded with zeroes.
        args: [u32; 4],
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
            | Self::Call1 { pc, .. }
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
        Insn::Call1 { helper, args, .. } => args[..helper.register_count()].iter().all(|&r| reg(r)),
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
#[expect(
    clippy::too_many_lines,
    reason = "Lowering keeps instruction control flow and helper exits together."
)]
pub(crate) fn emit(
    builder: &mut FunctionBuilder<'_>,
    module: &mut impl Module,
    step: FuncId,
    branch: FuncId,
    calls: [FuncId; HelperKind::COUNT],
    insns: &[Insn],
) {
    let pointer_type = module.target_config().pointer_type();
    let step = module.declare_func_in_func(step, builder.func);
    let branch = module.declare_func_in_func(branch, builder.func);
    let calls = calls.map(|id| module.declare_func_in_func(id, builder.func));

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let context = builder.block_params(entry)[0];
    let depth = builder.block_params(entry)[1];
    let frame = builder.block_params(entry)[2];

    let blocks: Vec<ir::Block> = insns.iter().map(|_| builder.create_block()).collect();
    let index: HashMap<u32, usize> = insns
        .iter()
        .enumerate()
        .map(|(i, insn)| (insn.pc(), i))
        .collect();
    // Falling off the end leaves the frame to the interpreter.
    let exit = builder.create_block();
    builder.append_block_param(exit, types::I32);
    let falloff = builder.create_block();
    let block_at = |pc: u32| index.get(&pc).map(|&i| blocks[i]);

    builder.ins().jump(blocks[0], &[]);

    for (i, insn) in insns.iter().enumerate() {
        builder.switch_to_block(blocks[i]);
        // Reload at every instruction, including the first one on entry. A helper
        // may have moved the stack before reaching this block.
        let regs = builder
            .ins()
            .load(pointer_type, MemFlagsData::trusted(), frame, 0);
        let argc = builder
            .ins()
            .load(types::I32, MemFlagsData::trusted(), frame, 8);
        let fall = blocks.get(i + 1).copied();
        let exit_code = |builder: &mut FunctionBuilder<'_>| {
            builder.ins().iconst(types::I32, i64::from(status::EXIT))
        };
        // Inline paths branch to successors directly; rejected cases use the same
        // helper lowering as Step and Branch.
        let insn = match *insn {
            Insn::Fast { pc, next, op } => {
                let slow = builder.create_block();
                Inline {
                    builder,
                    regs,
                    argc,
                    slow,
                }
                .emit_op(op, block_at(next).or(fall).unwrap_or(falloff));
                builder.switch_to_block(slow);
                Insn::Step { pc, next }
            }
            Insn::FastBranch {
                pc,
                next,
                target,
                cond,
            } => {
                let slow = builder.create_block();
                Inline {
                    builder,
                    regs,
                    argc,
                    slow,
                }
                .emit_cond(
                    cond,
                    block_at(next).unwrap_or(falloff),
                    block_at(target).unwrap_or(falloff),
                );
                builder.switch_to_block(slow);
                Insn::Branch { pc, next, target }
            }
            other => other,
        };
        match insn {
            Insn::Step { next, .. } | Insn::Call1 { next, .. } => {
                let next_value = builder.ins().iconst(types::I32, i64::from(next));
                let call = match insn {
                    Insn::Step { pc, .. } => {
                        let pc = builder.ins().iconst(types::I32, i64::from(pc));
                        builder.ins().call(step, &[context, pc, next_value, depth])
                    }
                    Insn::Call1 { helper, args, .. } => {
                        let [a, b, c, d] =
                            args.map(|arg| builder.ins().iconst(types::I32, i64::from(arg)));
                        builder.ins().call(
                            calls[helper.index()],
                            &[context, next_value, depth, a, b, c, d],
                        )
                    }
                    _ => unreachable!(),
                };
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
                let taken =
                    builder
                        .ins()
                        .icmp_imm_u(IntCC::Equal, result, i64::from(status::TAKEN));
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

    builder.switch_to_block(falloff);
    let code = builder.ins().iconst(types::I32, i64::from(status::EXIT));
    builder.ins().jump(exit, &[code.into()]);

    builder.switch_to_block(exit);
    let result = builder.block_params(exit)[0];
    builder.ins().return_(&[result]);
}

struct Number {
    is_int: ir::Value,
    is_numeric: ir::Value,
    int: ir::Value,
    float: ir::Value,
}

struct Inline<'a, 'f> {
    builder: &'a mut FunctionBuilder<'f>,
    regs: ir::Value,
    argc: ir::Value,
    slow: ir::Block,
}

impl Inline<'_, '_> {
    fn emit_op(&mut self, op: FastOp, next: ir::Block) {
        let dst = match op {
            FastOp::Move { dst, .. }
            | FastOp::Const { dst, .. }
            | FastOp::Arith { dst, .. }
            | FastOp::Inc { dst, .. }
            | FastOp::GetArg { dst, .. } => dst,
        };
        let old = self.load_reg(dst);
        self.reject_pointer(old);
        match op {
            FastOp::Move { src, .. } => {
                let value = self.load_reg(src);
                self.reject_pointer(value);
                self.store_and_go(dst, value, next);
            }
            FastOp::Const { bits, .. } => {
                let value = self.builder.ins().iconst(types::I64, imm_bits(bits));
                self.store_and_go(dst, value, next);
            }
            FastOp::Arith { op, lhs, rhs, .. } => {
                let left = self.load_reg(lhs);
                let right = self.load_reg(rhs);
                let left = self.number(left);
                let right = self.number(right);
                let floats = self.split_numbers(&left, &right);
                self.emit_arith(op, dst, &left, &right, floats, next);
            }
            FastOp::Inc { src, dec, .. } => {
                let value = self.load_reg(src);
                let value = self.number(value);
                self.guard(value.is_numeric);
                let one = self.builder.ins().iconst(types::I32, 1);
                let (integer, overflow) = if dec {
                    self.builder.ins().ssub_overflow(value.int, one)
                } else {
                    self.builder.ins().sadd_overflow(value.int, one)
                };
                let fits = self.builder.ins().icmp_imm_u(IntCC::Equal, overflow, 0);
                let use_int = self.builder.ins().band(value.is_int, fits);
                let one = self.fconst(1.0);
                let float = if dec {
                    self.builder.ins().fsub(value.float, one)
                } else {
                    self.builder.ins().fadd(value.float, one)
                };
                let integer = self.box_int(integer);
                let float = self.box_float(float);
                let result = self.builder.ins().select(use_int, integer, float);
                self.store_and_go(dst, result, next);
            }
            FastOp::GetArg { index, .. } => {
                self.emit_arg(index, dst, next);
            }
        }
    }

    fn emit_arg(&mut self, index: u32, dst: u32, next: ir::Block) {
        let present = self.builder.create_block();
        let absent = self.builder.create_block();
        let in_range =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::UnsignedGreaterThan, self.argc, i64::from(index));
        self.builder.ins().brif(in_range, present, &[], absent, &[]);

        self.builder.switch_to_block(present);
        let argc = self.builder.ins().uextend(types::I64, self.argc);
        let index = self.builder.ins().iconst(types::I64, i64::from(index));
        let distance = self.builder.ins().isub(argc, index);
        let bytes = self.builder.ins().ishl_imm_s(distance, 3);
        let address = self.builder.ins().isub(self.regs, bytes);
        let value = self
            .builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        self.reject_pointer(value);
        self.store_and_go(dst, value, next);

        self.builder.switch_to_block(absent);
        let value = self
            .builder
            .ins()
            .iconst(types::I64, imm_bits(boxed::UNDEFINED));
        self.store_and_go(dst, value, next);
    }

    fn emit_arith(
        &mut self,
        op: Arith,
        dst: u32,
        left: &Number,
        right: &Number,
        floats: ir::Block,
        next: ir::Block,
    ) {
        match op {
            Arith::Div => {
                self.emit_int_div(dst, left.int, right.int, floats, next);
            }
            Arith::Add | Arith::Sub | Arith::Mul => {
                let (integer, overflow) = match op {
                    Arith::Add => self.builder.ins().sadd_overflow(left.int, right.int),
                    Arith::Sub => self.builder.ins().ssub_overflow(left.int, right.int),
                    Arith::Mul => self.builder.ins().smul_overflow(left.int, right.int),
                    Arith::Div => unreachable!(),
                };
                let use_float = if matches!(op, Arith::Mul) {
                    let zero = self.builder.ins().icmp_imm_u(IntCC::Equal, integer, 0);
                    let left_negative =
                        self.builder
                            .ins()
                            .icmp_imm_u(IntCC::SignedLessThan, left.int, 0);
                    let right_negative =
                        self.builder
                            .ins()
                            .icmp_imm_u(IntCC::SignedLessThan, right.int, 0);
                    let negative = self.builder.ins().bor(left_negative, right_negative);
                    let negative_zero = self.builder.ins().band(zero, negative);
                    self.builder.ins().bor(overflow, negative_zero)
                } else {
                    overflow
                };
                let float = self.float_arith(op, left.float, right.float);
                let float = self.box_float(float);
                let integer = self.box_int(integer);
                let result = self.builder.ins().select(use_float, float, integer);
                self.store_and_go(dst, result, next);
            }
        }

        self.builder.switch_to_block(floats);
        let result = self.float_arith(op, left.float, right.float);
        let result = self.box_float(result);
        self.store_and_go(dst, result, next);
    }

    fn emit_int_div(
        &mut self,
        dst: u32,
        left: ir::Value,
        right: ir::Value,
        floats: ir::Block,
        next: ir::Block,
    ) {
        let zero = self.builder.ins().icmp_imm_u(IntCC::Equal, right, 0);
        let min = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::Equal, left, i64::from(i32::MIN));
        let minus_one = self.builder.ins().icmp_imm_u(IntCC::Equal, right, -1);
        let overflow = self.builder.ins().band(min, minus_one);
        let unsafe_division = self.builder.ins().bor(zero, overflow);
        let divide = self.builder.create_block();
        // Neither sdiv nor srem may execute for zero or MIN / -1.
        self.builder
            .ins()
            .brif(unsafe_division, floats, &[], divide, &[]);

        self.builder.switch_to_block(divide);
        let quotient = self.builder.ins().sdiv(left, right);
        let remainder = self.builder.ins().srem(left, right);
        let exact = self.builder.ins().icmp_imm_u(IntCC::Equal, remainder, 0);
        let left_zero = self.builder.ins().icmp_imm_u(IntCC::Equal, left, 0);
        let right_negative = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::SignedLessThan, right, 0);
        let negative_zero = self.builder.ins().band(left_zero, right_negative);
        let positive_zero = self
            .builder
            .ins()
            .icmp_imm_u(IntCC::Equal, negative_zero, 0);
        let use_int = self.builder.ins().band(exact, positive_zero);
        let integer = self.builder.create_block();
        self.builder.ins().brif(use_int, integer, &[], floats, &[]);

        self.builder.switch_to_block(integer);
        let result = self.box_int(quotient);
        self.store_and_go(dst, result, next);
    }

    fn emit_cond(&mut self, cond: FastCond, next: ir::Block, target: ir::Block) {
        match cond {
            FastCond::JumpUnless { rel, lhs, rhs } => {
                let left = self.load_reg(lhs);
                let right = self.load_reg(rhs);
                let left = self.number(left);
                let right = self.number(right);
                let floats = self.split_numbers(&left, &right);
                let cc = match rel {
                    Rel::Lt => IntCC::SignedLessThan,
                    Rel::Le => IntCC::SignedLessThanOrEqual,
                    Rel::Gt => IntCC::SignedGreaterThan,
                    Rel::Ge => IntCC::SignedGreaterThanOrEqual,
                };
                let holds = self.builder.ins().icmp(cc, left.int, right.int);
                self.builder.ins().brif(holds, next, &[], target, &[]);

                self.builder.switch_to_block(floats);
                let cc = match rel {
                    Rel::Lt => FloatCC::LessThan,
                    Rel::Le => FloatCC::LessThanOrEqual,
                    Rel::Gt => FloatCC::GreaterThan,
                    Rel::Ge => FloatCC::GreaterThanOrEqual,
                };
                // Ordered relations are false whenever either operand is NaN.
                let holds = self.builder.ins().fcmp(cc, left.float, right.float);
                self.builder.ins().brif(holds, next, &[], target, &[]);
            }
            FastCond::JumpIf { when_true, value } => {
                let value = self.load_reg(value);
                let number = self.number(value);
                let boolean = self.kind_is(value, boxed::KIND_BOOLEAN);
                let nullish = self.kind_is(value, boxed::UNDEFINED & boxed::MASK_KIND);
                let scalar = self.builder.ins().bor(number.is_numeric, boolean);
                let supported = self.builder.ins().bor(scalar, nullish);
                // Pointer kinds and unsupported tags must use the branch helper.
                self.guard(supported);

                let bit = self.builder.ins().band_imm_s(value, 1);
                let bool_truth = self.builder.ins().icmp_imm_u(IntCC::NotEqual, bit, 0);
                let int_truth = self
                    .builder
                    .ins()
                    .icmp_imm_u(IntCC::NotEqual, number.int, 0);
                let zero = self.fconst(0.0);
                let nonzero = self
                    .builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, number.float, zero);
                let ordered = self
                    .builder
                    .ins()
                    .fcmp(FloatCC::Ordered, number.float, number.float);
                let float_truth = self.builder.ins().band(nonzero, ordered);
                let num_truth = self
                    .builder
                    .ins()
                    .select(number.is_int, int_truth, float_truth);
                let truth = self.builder.ins().select(boolean, bool_truth, num_truth);
                let truth = self.builder.ins().band(scalar, truth);
                let (yes, no) = if when_true {
                    (target, next)
                } else {
                    (next, target)
                };
                self.builder.ins().brif(truth, yes, &[], no, &[]);
            }
        }
    }

    fn reg_address(&mut self, reg: u32) -> ir::Value {
        self.builder.ins().iadd_imm_s(self.regs, i64::from(reg) * 8)
    }

    fn load_reg(&mut self, reg: u32) -> ir::Value {
        let address = self.reg_address(reg);
        self.builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0)
    }

    fn store_and_go(&mut self, dst: u32, value: ir::Value, next: ir::Block) {
        let address = self.reg_address(dst);
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), value, address, 0);
        self.builder.ins().jump(next, &[]);
    }

    fn guard(&mut self, condition: ir::Value) {
        let fast = self.builder.create_block();
        self.builder
            .ins()
            .brif(condition, fast, &[], self.slow, &[]);
        self.builder.switch_to_block(fast);
    }

    fn reject_pointer(&mut self, value: ir::Value) {
        let masked = self
            .builder
            .ins()
            .band_imm_s(value, imm_bits(boxed::MASK_POINTER));
        let safe =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::NotEqual, masked, imm_bits(boxed::MASK_POINTER));
        self.guard(safe);
    }

    fn kind_is(&mut self, value: ir::Value, kind: u64) -> ir::Value {
        let masked = self
            .builder
            .ins()
            .band_imm_s(value, imm_bits(boxed::MASK_KIND));
        self.builder
            .ins()
            .icmp_imm_u(IntCC::Equal, masked, imm_bits(kind))
    }

    fn number(&mut self, value: ir::Value) -> Number {
        let is_int = self.kind_is(value, boxed::KIND_INT32);
        let exponent = self
            .builder
            .ins()
            .band_imm_s(value, imm_bits(boxed::MASK_NAN));
        let finite =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::NotEqual, exponent, imm_bits(boxed::MASK_NAN));
        let infinity = self.kind_is(value, boxed::KIND_INFINITY);
        let nan = self.kind_is(value, boxed::CANONICAL_NAN & boxed::MASK_KIND);
        let special_float = self.builder.ins().bor(infinity, nan);
        let is_float = self.builder.ins().bor(finite, special_float);
        let is_numeric = self.builder.ins().bor(is_int, is_float);
        let int = self.builder.ins().ireduce(types::I32, value);
        let converted = self.builder.ins().fcvt_from_sint(types::F64, int);
        let raw = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), value);
        let float = self.builder.ins().select(is_int, converted, raw);
        Number {
            is_int,
            is_numeric,
            int,
            float,
        }
    }

    /// Leaves the builder in the both-int32 block and returns the float block.
    fn split_numbers(&mut self, left: &Number, right: &Number) -> ir::Block {
        let numbers = self.builder.ins().band(left.is_numeric, right.is_numeric);
        self.guard(numbers);
        let integers = self.builder.ins().band(left.is_int, right.is_int);
        let ints = self.builder.create_block();
        let floats = self.builder.create_block();
        self.builder.ins().brif(integers, ints, &[], floats, &[]);
        self.builder.switch_to_block(ints);
        floats
    }

    fn box_int(&mut self, value: ir::Value) -> ir::Value {
        let value = self.builder.ins().uextend(types::I64, value);
        self.builder
            .ins()
            .bor_imm_s(value, imm_bits(boxed::KIND_INT32))
    }

    fn box_float(&mut self, value: ir::Value) -> ir::Value {
        let nan = self.builder.ins().fcmp(FloatCC::Unordered, value, value);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), value);
        let canonical = self
            .builder
            .ins()
            .iconst(types::I64, imm_bits(boxed::CANONICAL_NAN));
        self.builder.ins().select(nan, canonical, bits)
    }

    fn fconst(&mut self, value: f64) -> ir::Value {
        self.builder.ins().f64const(Ieee64::with_float(value))
    }

    fn float_arith(&mut self, op: Arith, left: ir::Value, right: ir::Value) -> ir::Value {
        match op {
            Arith::Add => self.builder.ins().fadd(left, right),
            Arith::Sub => self.builder.ins().fsub(left, right),
            Arith::Mul => self.builder.ins().fmul(left, right),
            Arith::Div => self.builder.ins().fdiv(left, right),
        }
    }
}

/// Interprets a boxed bit pattern as a signed Cranelift immediate without changing it.
fn imm_bits(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}
