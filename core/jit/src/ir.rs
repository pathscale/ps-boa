//! The numeric IR [`crate::compile_numeric_leaf`] lowers.
//!
//! The engine translates a function body into a list of [`Insn`] in one forward pass
//! over its bytecode, keeping each instruction's bytecode address as `pc`; jump targets
//! name those addresses. Registers are the engine's register indices. Every value is a
//! JavaScript Number held as f64, except comparison results, which are booleans.

use std::collections::HashSet;

/// `i32::MAX / 8`. A parameter index times 8 must fit in a Cranelift load offset.
const MAX_PARAMETERS: u32 = 268_435_455;

/// A two-operand Number operation, with JavaScript semantics.
#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    /// `+` on Numbers.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
    /// `%`: the remainder takes the sign of the dividend.
    Mod,
    /// `&` after `ToInt32` of both sides.
    BitAnd,
    /// `|`.
    BitOr,
    /// `^`.
    BitXor,
    /// `<<`: the shift count is `ToUint32(rhs) & 31`.
    Shl,
    /// `>>`.
    Shr,
    /// `>>>`: the result is unsigned.
    Ushr,
}

/// A comparison producing a boolean.
#[derive(Debug, Clone, Copy)]
pub enum CmpOp {
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `===`.
    StrictEq,
    /// `!==`.
    StrictNe,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
}

/// The relation a fused compare-and-jump tests. The jump is taken when it does not hold.
#[derive(Debug, Clone, Copy)]
pub enum Rel {
    /// Jump unless `lhs < rhs`.
    Lt,
    /// Jump unless `lhs <= rhs`.
    Le,
    /// Jump unless `lhs > rhs`.
    Gt,
    /// Jump unless `lhs >= rhs`.
    Ge,
}

/// Which truthiness a conditional jump takes.
#[derive(Debug, Clone, Copy)]
pub enum When {
    /// Jump when the value is truthy.
    True,
    /// Jump when the value is falsy.
    False,
}

/// An in-place unary Number operation.
#[derive(Debug, Clone, Copy)]
pub enum Unary {
    /// Unary `+`.
    Pos,
    /// Unary `-`.
    Neg,
}

/// A pure `Math` function computed inline. Each is exact IEEE arithmetic, so the result is
/// the one the interpreter's intrinsic returns.
#[derive(Debug, Clone, Copy)]
pub enum MathFn {
    /// `Math.floor`.
    Floor,
    /// `Math.ceil`.
    Ceil,
    /// `Math.trunc`.
    Trunc,
    /// `Math.sqrt`.
    Sqrt,
    /// `Math.abs`.
    Abs,
    /// `Math.max`: `NaN` if any argument is, and +0 over -0.
    Max,
    /// `Math.min`: `NaN` if any argument is, and -0 over +0.
    Min,
}

/// One instruction. Register operands are engine register indices.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// `dst = value`.
    Const {
        /// Destination register.
        dst: u32,
        /// The Number stored.
        value: f64,
    },
    /// `dst = src`.
    Move {
        /// Destination register.
        dst: u32,
        /// Source register.
        src: u32,
    },
    /// `dst = arguments[index]`.
    GetArg {
        /// Formal parameter index.
        index: u32,
        /// Destination register.
        dst: u32,
    },
    /// `reg = op reg`.
    Unary {
        /// The operation.
        op: Unary,
        /// Operand and destination register.
        reg: u32,
    },
    /// `reg = ~reg`.
    BitNot {
        /// Operand and destination register.
        reg: u32,
    },
    /// `dst = src + 1`, or `src - 1` when `dec`.
    Inc {
        /// Destination register.
        dst: u32,
        /// Source register.
        src: u32,
        /// Decrement instead of increment.
        dec: bool,
    },
    /// `dst = lhs op rhs`.
    Bin {
        /// The operation.
        op: BinOp,
        /// Destination register.
        dst: u32,
        /// Left operand register.
        lhs: u32,
        /// Right operand register.
        rhs: u32,
    },
    /// `dst = lhs op rhs`, a boolean.
    Cmp {
        /// The comparison.
        op: CmpOp,
        /// Destination register.
        dst: u32,
        /// Left operand register.
        lhs: u32,
        /// Right operand register.
        rhs: u32,
    },
    /// Unconditional jump.
    Jump {
        /// Target `pc`.
        target: u32,
    },
    /// Jump on the truthiness of `value`.
    JumpIf {
        /// Which truthiness jumps.
        when: When,
        /// Target `pc`.
        target: u32,
        /// Tested register.
        value: u32,
    },
    /// Jump unless `lhs rel rhs`.
    JumpRel {
        /// The relation.
        rel: Rel,
        /// Target `pc`.
        target: u32,
        /// Left operand register.
        lhs: u32,
        /// Right operand register.
        rhs: u32,
    },
    /// Jump unless `lhs == rhs`.
    JumpNe {
        /// Target `pc`.
        target: u32,
        /// Left operand register.
        lhs: u32,
        /// Right operand register.
        rhs: u32,
    },
    /// `dst = Math.f(args[..argc])`. Valid only while the realm's `Math` is untouched,
    /// which the engine checks before running the code.
    Math {
        /// The function.
        f: MathFn,
        /// Destination register.
        dst: u32,
        /// Argument registers; the first `argc` are live.
        args: [u32; 3],
        /// Argument count, at most 3.
        argc: u8,
    },
    /// No effect.
    Nop,
    /// `accumulator = src`.
    SetAcc {
        /// Source register.
        src: u32,
    },
    /// No effect for an ordinary function: the accumulator is the return value.
    CheckReturn,
    /// Return the accumulator.
    Return,
    /// A loop back edge. No effect on the result.
    LoopIter,
}

impl Op {
    /// The `pc` this instruction can jump to.
    #[must_use]
    pub fn jump_target(self) -> Option<u32> {
        match self {
            Self::Jump { target }
            | Self::JumpIf { target, .. }
            | Self::JumpRel { target, .. }
            | Self::JumpNe { target, .. } => Some(target),
            _ => None,
        }
    }

    /// `true` when control never falls through to the next instruction unconditionally.
    #[must_use]
    pub fn is_terminator(self) -> bool {
        matches!(
            self,
            Self::Jump { .. }
                | Self::JumpIf { .. }
                | Self::JumpRel { .. }
                | Self::JumpNe { .. }
                | Self::Return
        )
    }
}

/// An [`Op`] at its bytecode address.
#[derive(Debug, Clone, Copy)]
pub struct Insn {
    /// Bytecode address of the instruction.
    pub pc: u32,
    /// The operation.
    pub op: Op,
}

/// `true` when every register is below `register_count`, every argument index is below
/// `parameter_count`, and every jump target is the `pc` of an instruction.
pub(crate) fn validate(insns: &[Insn], register_count: u32, parameter_count: u32) -> bool {
    if insns.is_empty() || parameter_count > MAX_PARAMETERS {
        return false;
    }
    let reg = |r: u32| r < register_count;
    let starts: HashSet<u32> = insns.iter().map(|insn| insn.pc).collect();
    insns.iter().all(|insn| {
        let registers = match insn.op {
            Op::Const { dst, .. } => reg(dst),
            Op::Move { dst, src } | Op::Inc { dst, src, .. } => reg(dst) && reg(src),
            Op::GetArg { index, dst } => index < parameter_count && reg(dst),
            Op::Unary { reg: r, .. } | Op::BitNot { reg: r } => reg(r),
            Op::Bin { dst, lhs, rhs, .. } | Op::Cmp { dst, lhs, rhs, .. } => {
                reg(dst) && reg(lhs) && reg(rhs)
            }
            Op::JumpIf { value, .. } => reg(value),
            Op::JumpRel { lhs, rhs, .. } | Op::JumpNe { lhs, rhs, .. } => reg(lhs) && reg(rhs),
            Op::SetAcc { src } => reg(src),
            Op::Math {
                dst, args, argc, ..
            } => argc <= 3 && reg(dst) && args[..usize::from(argc)].iter().all(|&a| reg(a)),
            Op::Jump { .. } | Op::CheckReturn | Op::Return | Op::LoopIter | Op::Nop => true,
        };
        registers
            && insn
                .op
                .jump_target()
                .is_none_or(|target| starts.contains(&target))
    })
}
