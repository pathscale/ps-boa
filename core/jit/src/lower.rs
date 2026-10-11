//! Kind analysis and Cranelift emission for a decoded numeric leaf.
//!
//! Booleans are f64 0.0 or 1.0 plus a kind tag, so `ToNumber` of a comparison
//! result is the same bits and `ToBoolean` matches both numbers and booleans.
//! A `SetAccumulator` of a boolean rejects the function: the return would be a
//! number, not a boolean.

use crate::ir::{BinOp, CmpOp, Insn, MathFn, Op, Rel, Unary, When, validate};
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::immediates::Ieee64;
use cranelift_codegen::ir::{self, InstBuilder, MemFlagsData, Type, Value, types};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::{FuncId, Module};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Undef,
    Num,
    Bool,
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    regs: Vec<Kind>,
    acc: Kind,
}

pub(crate) struct Prepared {
    insns: Vec<Insn>,
    regions: Vec<Region>,
    register_count: usize,
}

struct Region {
    range_start: usize,
    range_end: usize,
    target: Option<usize>,
    fall: Option<usize>,
    falloff: bool,
    reachable: bool,
    inbound: State,
}

pub(crate) fn prepare(
    insns: Vec<Insn>,
    register_count: u32,
    parameter_count: u32,
) -> Option<Prepared> {
    if !validate(&insns, register_count, parameter_count) {
        return None;
    }
    let mut regions = build_regions(&insns)?;
    mark_reachable(&mut regions)?;
    analyse(&insns, &mut regions, register_count)?;
    Some(Prepared {
        insns,
        regions,
        register_count: (register_count as usize),
    })
}

pub(crate) fn emit(
    builder: &mut FunctionBuilder<'_>,
    module: &mut impl Module,
    rem: FuncId,
    to_int32: FuncId,
    pointer_type: Type,
    prepared: &Prepared,
) {
    let mut regs = Vec::with_capacity(prepared.register_count);
    for _ in 0..prepared.register_count {
        regs.push(builder.declare_var(types::F64));
    }
    let acc = builder.declare_var(types::F64);
    let args = builder.declare_var(pointer_type);
    let rem = module.declare_func_in_func(rem, builder.func);
    let to_int32 = module.declare_func_in_func(to_int32, builder.func);

    let mut clif = Vec::with_capacity(prepared.regions.len());
    for region in &prepared.regions {
        if region.reachable {
            clif.push(Some(builder.create_block()));
        } else {
            clif.push(None);
        }
    }

    let mut cx = Cx {
        builder,
        regs: &regs,
        acc,
        args,
        rem,
        to_int32,
    };
    for (index, region) in prepared.regions.iter().enumerate() {
        if !region.reachable {
            continue;
        }
        let Some(block) = clif[index] else {
            continue;
        };
        cx.builder.switch_to_block(block);
        if index == 0 {
            cx.builder.append_block_params_for_function_params(block);
            let Some(param) = cx.builder.block_params(block).first().copied() else {
                cx.ret0();
                continue;
            };
            cx.builder.def_var(cx.args, param);
        } else {
            let ptr = cx.builder.use_var(cx.args);
            cx.builder.def_var(cx.args, ptr);
        }
        cx.refresh(region);
        if region.range_start >= region.range_end {
            cx.ret0();
            continue;
        }
        let mut state = region.inbound.clone();
        let last = region.range_end - 1;
        for insn in region.range_start..last {
            cx.emit_value(&mut state, &prepared.insns[insn].op);
        }
        let target = resolve(&clif, region.target);
        let fall = resolve(&clif, region.fall);
        cx.emit_last(&mut state, &prepared.insns[last].op, target, fall);
    }
}

fn resolve(clif: &[Option<ir::Block>], index: Option<usize>) -> Option<ir::Block> {
    index.and_then(|index| clif.get(index).copied().flatten())
}

struct Cx<'a, 'f, 'r> {
    builder: &'a mut FunctionBuilder<'f>,
    regs: &'r [Variable],
    acc: Variable,
    args: Variable,
    rem: ir::FuncRef,
    to_int32: ir::FuncRef,
}

impl Cx<'_, '_, '_> {
    fn refresh(&mut self, region: &Region) {
        for (index, kind) in region.inbound.regs.iter().enumerate() {
            if *kind == Kind::Undef {
                continue;
            }
            let value = self.builder.use_var(self.regs[index]);
            self.builder.def_var(self.regs[index], value);
        }
        if region.inbound.acc != Kind::Undef {
            let value = self.builder.use_var(self.acc);
            self.builder.def_var(self.acc, value);
        }
    }

    fn emit_value(&mut self, state: &mut State, op: &Op) {
        match *op {
            Op::Const { dst, value } => {
                let _ = apply(state, op);
                let value = self.fconst(value);
                self.def_reg(dst, value);
            }
            Op::Move { dst, src } => {
                let _ = apply(state, op);
                let value = self.use_reg(src);
                self.def_reg(dst, value);
            }
            Op::GetArg { index, dst } => {
                let _ = apply(state, op);
                let ptr = self.builder.use_var(self.args);
                let offset = i32::try_from(u64::from(index) * 8).unwrap_or(0);
                let value =
                    self.builder
                        .ins()
                        .load(types::F64, MemFlagsData::trusted(), ptr, offset);
                self.def_reg(dst, value);
            }
            Op::Unary { op: unary, reg } => {
                let _ = apply(state, op);
                let value = self.use_reg(reg);
                let value = match unary {
                    Unary::Pos => value,
                    Unary::Neg => self.builder.ins().fneg(value),
                };
                self.def_reg(reg, value);
            }
            Op::BitNot { reg } => {
                let _ = apply(state, op);
                let value = self.use_reg(reg);
                let bits = self.call_i32(value);
                let bits = self.builder.ins().bnot(bits);
                let value = self.i32_signed(bits);
                self.def_reg(reg, value);
            }
            Op::Inc { dst, src, dec } => {
                let _ = apply(state, op);
                let value = self.use_reg(src);
                self.def_reg(src, value);
                let one = self.fconst(1.0);
                let updated = if dec {
                    self.builder.ins().fsub(value, one)
                } else {
                    self.builder.ins().fadd(value, one)
                };
                self.def_reg(dst, updated);
            }
            Op::Bin {
                op: bin,
                dst,
                lhs,
                rhs,
            } => {
                let _ = apply(state, op);
                let left = self.use_reg(lhs);
                let right = self.use_reg(rhs);
                let value = self.bin_value(bin, left, right);
                self.def_reg(dst, value);
            }
            Op::Cmp {
                op: cmp,
                dst,
                lhs,
                rhs,
            } => {
                let left_kind = state.regs[lhs as usize];
                let right_kind = state.regs[rhs as usize];
                let _ = apply(state, op);
                let left = self.use_reg(lhs);
                let right = self.use_reg(rhs);
                let value = self.cmp_value(cmp, left, right, left_kind, right_kind);
                self.def_reg(dst, value);
            }
            Op::SetAcc { src } => {
                let _ = apply(state, op);
                let value = self.use_reg(src);
                self.builder.def_var(self.acc, value);
            }
            Op::Math { f, dst, args, argc } => {
                let _ = apply(state, op);
                let value = self.math_value(f, &args[..usize::from(argc)]);
                self.def_reg(dst, value);
            }
            Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::JumpRel { .. }
            | Op::JumpNe { .. }
            | Op::CheckReturn
            | Op::Return
            | Op::LoopIter
            | Op::Nop => {
                let _ = apply(state, op);
            }
        }
    }

    /// A boolean argument is 0.0 or 1.0, which is already its `ToNumber`.
    fn math_value(&mut self, f: MathFn, regs: &[u32]) -> Value {
        let args: Vec<Value> = regs.iter().map(|&reg| self.use_reg(reg)).collect();
        let unary = |cx: &mut Self, op: fn(&mut Self, Value) -> Value| match args.first() {
            Some(&value) => op(cx, value),
            None => cx.fconst(f64::NAN),
        };
        match f {
            MathFn::Floor => unary(self, |cx, v| cx.builder.ins().floor(v)),
            MathFn::Ceil => unary(self, |cx, v| cx.builder.ins().ceil(v)),
            MathFn::Trunc => unary(self, |cx, v| cx.builder.ins().trunc(v)),
            MathFn::Sqrt => unary(self, |cx, v| cx.builder.ins().sqrt(v)),
            MathFn::Abs => unary(self, |cx, v| cx.builder.ins().fabs(v)),
            MathFn::Max | MathFn::Min => {
                let max = matches!(f, MathFn::Max);
                let mut result = self.fconst(if max {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                });
                for &value in &args {
                    result = if max {
                        self.builder.ins().fmax(result, value)
                    } else {
                        self.builder.ins().fmin(result, value)
                    };
                }
                result
            }
        }
    }

    fn emit_last(
        &mut self,
        state: &mut State,
        op: &Op,
        target: Option<ir::Block>,
        fall: Option<ir::Block>,
    ) {
        match *op {
            Op::Jump { .. } => {
                let _ = apply(state, op);
                self.go(target);
            }
            Op::JumpIf { when, value, .. } => {
                let _ = apply(state, op);
                let value = self.use_reg(value);
                let cond = self.truthy(value);
                let (then_block, else_block) = match when {
                    When::True => (target, fall),
                    When::False => (fall, target),
                };
                self.brif(cond, then_block, else_block);
            }
            Op::JumpRel { rel, lhs, rhs, .. } => {
                let _ = apply(state, op);
                let left = self.use_reg(lhs);
                let right = self.use_reg(rhs);
                let cond = self.builder.ins().fcmp(rel_cc(rel), left, right);
                // The relation holds: fall through. Otherwise take the target.
                self.brif(cond, fall, target);
            }
            Op::JumpNe { lhs, rhs, .. } => {
                let left_kind = state.regs[lhs as usize];
                let right_kind = state.regs[rhs as usize];
                let _ = apply(state, op);
                if left_kind != right_kind {
                    self.go(target);
                    return;
                }
                let left = self.use_reg(lhs);
                let right = self.use_reg(rhs);
                let equal = self.same_value_zero(left, right);
                self.brif(equal, fall, target);
            }
            Op::Return => {
                let _ = apply(state, op);
                let value = self.builder.use_var(self.acc);
                self.builder.ins().return_(&[value]);
            }
            _ => {
                self.emit_value(state, op);
                self.go(fall);
            }
        }
    }

    fn bin_value(&mut self, op: BinOp, left: Value, right: Value) -> Value {
        match op {
            BinOp::Add => self.builder.ins().fadd(left, right),
            BinOp::Sub => self.builder.ins().fsub(left, right),
            BinOp::Mul => self.builder.ins().fmul(left, right),
            BinOp::Div => self.builder.ins().fdiv(left, right),
            BinOp::Mod => self.call_f64(self.rem, &[left, right]),
            BinOp::BitAnd => self.bit_signed(left, right, |cx, a, b| cx.builder.ins().band(a, b)),
            BinOp::BitOr => self.bit_signed(left, right, |cx, a, b| cx.builder.ins().bor(a, b)),
            BinOp::BitXor => self.bit_signed(left, right, |cx, a, b| cx.builder.ins().bxor(a, b)),
            BinOp::Shl => self.bit_signed(left, right, |cx, a, b| cx.builder.ins().ishl(a, b)),
            BinOp::Shr => self.bit_signed(left, right, |cx, a, b| cx.builder.ins().sshr(a, b)),
            BinOp::Ushr => {
                let left = self.call_i32(left);
                let right = self.call_i32(right);
                let bits = self.builder.ins().ushr(left, right);
                self.builder.ins().fcvt_from_uint(types::F64, bits)
            }
        }
    }

    fn bit_signed(
        &mut self,
        left: Value,
        right: Value,
        op: impl FnOnce(&mut Self, Value, Value) -> Value,
    ) -> Value {
        let left = self.call_i32(left);
        let right = self.call_i32(right);
        let bits = op(self, left, right);
        self.i32_signed(bits)
    }

    fn cmp_value(
        &mut self,
        op: CmpOp,
        left: Value,
        right: Value,
        left_kind: Kind,
        right_kind: Kind,
    ) -> Value {
        match op {
            CmpOp::StrictEq if left_kind != right_kind => self.fconst(0.0),
            CmpOp::StrictNe if left_kind != right_kind => self.fconst(1.0),
            CmpOp::Eq | CmpOp::StrictEq => {
                let cond = self.builder.ins().fcmp(FloatCC::Equal, left, right);
                self.bool_from_cond(cond)
            }
            CmpOp::Ne | CmpOp::StrictNe => {
                let cond = self.builder.ins().fcmp(FloatCC::NotEqual, left, right);
                self.bool_from_cond(cond)
            }
            CmpOp::Lt => self.rel_bool(FloatCC::LessThan, left, right),
            CmpOp::Le => self.rel_bool(FloatCC::LessThanOrEqual, left, right),
            CmpOp::Gt => self.rel_bool(FloatCC::GreaterThan, left, right),
            CmpOp::Ge => self.rel_bool(FloatCC::GreaterThanOrEqual, left, right),
        }
    }

    fn rel_bool(&mut self, cc: FloatCC, left: Value, right: Value) -> Value {
        let cond = self.builder.ins().fcmp(cc, left, right);
        self.bool_from_cond(cond)
    }

    fn brif(&mut self, cond: Value, then_block: Option<ir::Block>, else_block: Option<ir::Block>) {
        match (then_block, else_block) {
            (Some(then_block), Some(else_block)) => {
                self.builder
                    .ins()
                    .brif(cond, then_block, &[], else_block, &[]);
            }
            (Some(only), None) | (None, Some(only)) => {
                self.builder.ins().jump(only, &[]);
            }
            (None, None) => self.ret0(),
        }
    }

    fn go(&mut self, block: Option<ir::Block>) {
        match block {
            Some(block) => {
                self.builder.ins().jump(block, &[]);
            }
            None => self.ret0(),
        }
    }

    fn ret0(&mut self) {
        let zero = self.fconst(0.0);
        self.builder.ins().return_(&[zero]);
    }

    fn use_reg(&mut self, index: u32) -> Value {
        self.builder.use_var(self.regs[index as usize])
    }

    fn def_reg(&mut self, index: u32, value: Value) {
        self.builder.def_var(self.regs[index as usize], value);
    }

    fn fconst(&mut self, value: f64) -> Value {
        self.builder.ins().f64const(Ieee64::with_float(value))
    }

    fn bool_from_cond(&mut self, cond: Value) -> Value {
        let one = self.fconst(1.0);
        let zero = self.fconst(0.0);
        self.builder.ins().select(cond, one, zero)
    }

    fn truthy(&mut self, value: Value) -> Value {
        let zero = self.fconst(0.0);
        let one = self.fconst(1.0);
        let ordered = self.builder.ins().fcmp(FloatCC::Ordered, value, value);
        let nonzero = self.builder.ins().fcmp(FloatCC::NotEqual, value, zero);
        let false_bit = self.builder.ins().fcmp(FloatCC::Equal, zero, one);
        self.builder.ins().select(ordered, nonzero, false_bit)
    }

    fn same_value_zero(&mut self, left: Value, right: Value) -> Value {
        let zero = self.fconst(0.0);
        let one = self.fconst(1.0);
        let equal = self.builder.ins().fcmp(FloatCC::Equal, left, right);
        let left_nan = self.builder.ins().fcmp(FloatCC::Unordered, left, left);
        let right_nan = self.builder.ins().fcmp(FloatCC::Unordered, right, right);
        let false_bit = self.builder.ins().fcmp(FloatCC::Equal, zero, one);
        let true_bit = self.builder.ins().fcmp(FloatCC::Equal, zero, zero);
        let both_nan = self.builder.ins().select(left_nan, right_nan, false_bit);
        self.builder.ins().select(equal, true_bit, both_nan)
    }

    fn call_i32(&mut self, value: Value) -> Value {
        let inst = self.builder.ins().call(self.to_int32, &[value]);
        self.builder
            .inst_results(inst)
            .first()
            .copied()
            .unwrap_or_else(|| self.builder.ins().iconst(types::I32, 0))
    }

    fn call_f64(&mut self, func: ir::FuncRef, args: &[Value]) -> Value {
        let inst = self.builder.ins().call(func, args);
        match self.builder.inst_results(inst).first().copied() {
            Some(value) => value,
            None => self.fconst(0.0),
        }
    }

    fn i32_signed(&mut self, bits: Value) -> Value {
        self.builder.ins().fcvt_from_sint(types::F64, bits)
    }
}

fn rel_cc(rel: Rel) -> FloatCC {
    match rel {
        Rel::Lt => FloatCC::LessThan,
        Rel::Le => FloatCC::LessThanOrEqual,
        Rel::Gt => FloatCC::GreaterThan,
        Rel::Ge => FloatCC::GreaterThanOrEqual,
    }
}

fn build_regions(insns: &[Insn]) -> Option<Vec<Region>> {
    if insns.is_empty() {
        return None;
    }
    let mut pc_to_insn = HashMap::with_capacity(insns.len());
    for (index, insn) in insns.iter().enumerate() {
        pc_to_insn.insert(insn.pc, index);
    }
    let mut leader = vec![false; insns.len()];
    leader[0] = true;
    for (index, insn) in insns.iter().enumerate() {
        if let Some(target) = insn.op.jump_target() {
            let target_index = *pc_to_insn.get(&target)?;
            leader[target_index] = true;
        }
        if insn.op.is_terminator() && index + 1 < insns.len() {
            leader[index + 1] = true;
        }
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, &is_leader) in leader.iter().enumerate().take(insns.len()).skip(1) {
        if is_leader {
            ranges.push((start, index));
            start = index;
        }
    }
    ranges.push((start, insns.len()));

    let mut block_of = vec![0usize; insns.len()];
    for (block, (start, end)) in ranges.iter().enumerate() {
        block_of[*start..*end].fill(block);
    }

    let mut regions = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let last = end - 1;
        let mut target = None;
        let mut fall = None;
        let mut falloff = false;
        match insns[last].op {
            Op::Jump { target: pc } => {
                let insn_index = *pc_to_insn.get(&pc)?;
                target = Some(block_of[insn_index]);
            }
            Op::Return => {}
            Op::JumpIf { target: pc, .. }
            | Op::JumpRel { target: pc, .. }
            | Op::JumpNe { target: pc, .. } => {
                let insn_index = *pc_to_insn.get(&pc)?;
                target = Some(block_of[insn_index]);
                if end < insns.len() {
                    fall = Some(block_of[end]);
                } else {
                    falloff = true;
                }
            }
            _ => {
                if end < insns.len() {
                    fall = Some(block_of[end]);
                } else {
                    falloff = true;
                }
            }
        }
        regions.push(Region {
            range_start: start,
            range_end: end,
            target,
            fall,
            falloff,
            reachable: false,
            inbound: State {
                regs: Vec::new(),
                acc: Kind::Undef,
            },
        });
    }
    Some(regions)
}

fn mark_reachable(regions: &mut [Region]) -> Option<()> {
    if regions.is_empty() {
        return None;
    }
    regions[0].reachable = true;
    let mut stack = vec![0usize];
    while let Some(block) = stack.pop() {
        let falloff = regions[block].falloff;
        let target = regions[block].target;
        let fall = regions[block].fall;
        if falloff {
            return None;
        }
        for succ in [target, fall].into_iter().flatten() {
            if !regions[succ].reachable {
                regions[succ].reachable = true;
                stack.push(succ);
            }
        }
    }
    Some(())
}

fn analyse(insns: &[Insn], regions: &mut [Region], register_count: u32) -> Option<()> {
    let undef = State {
        regs: vec![Kind::Undef; register_count as usize],
        acc: Kind::Undef,
    };
    let mut inbound = vec![None; regions.len()];
    if inbound.is_empty() {
        return None;
    }
    inbound[0] = Some(undef);
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    let mut queued = vec![false; regions.len()];
    queued[0] = true;
    let height = (register_count as usize).saturating_add(2);
    let limit = regions
        .len()
        .saturating_mul(height)
        .saturating_mul(4)
        .saturating_add(8);
    let mut steps = 0usize;
    while let Some(block) = queue.pop_front() {
        queued[block] = false;
        if !regions[block].reachable {
            continue;
        }
        steps += 1;
        if steps > limit {
            return None;
        }
        let mut state = inbound[block].clone()?;
        let start = regions[block].range_start;
        let end = regions[block].range_end;
        for insn in &insns[start..end] {
            if !apply(&mut state, &insn.op) {
                return None;
            }
        }
        let succs = [regions[block].target, regions[block].fall];
        for succ in succs.into_iter().flatten() {
            let joined = match &inbound[succ] {
                None => Some(state.clone()),
                Some(prev) => join_state(prev, &state),
            };
            let joined = joined?;
            if inbound[succ].as_ref() != Some(&joined) {
                inbound[succ] = Some(joined);
                if !queued[succ] {
                    queued[succ] = true;
                    queue.push_back(succ);
                }
            }
        }
    }
    for (region, entry) in regions.iter_mut().zip(inbound) {
        if region.reachable {
            region.inbound = entry?;
        }
    }
    Some(())
}

fn join_state(left: &State, right: &State) -> Option<State> {
    let mut regs = Vec::with_capacity(left.regs.len());
    for (left, right) in left.regs.iter().zip(right.regs.iter()) {
        regs.push(join_kind(*left, *right)?);
    }
    Some(State {
        regs,
        acc: join_kind(left.acc, right.acc)?,
    })
}

fn join_kind(left: Kind, right: Kind) -> Option<Kind> {
    match (left, right) {
        (Kind::Num, Kind::Num) => Some(Kind::Num),
        (Kind::Bool, Kind::Bool) => Some(Kind::Bool),
        (Kind::Undef, _) | (_, Kind::Undef) => Some(Kind::Undef),
        (Kind::Num, Kind::Bool) | (Kind::Bool, Kind::Num) => None,
    }
}

fn apply(state: &mut State, op: &Op) -> bool {
    match *op {
        Op::Const { dst, .. } | Op::GetArg { dst, .. } => set_kind(state, dst, Kind::Num),
        Op::Move { dst, src } => {
            if !defined(state, src) {
                return false;
            }
            let kind = kind_of(state, src);
            set_kind(state, dst, kind);
        }
        Op::Unary { reg, .. } | Op::BitNot { reg } => {
            if !defined(state, reg) {
                return false;
            }
            set_kind(state, reg, Kind::Num);
        }
        Op::Inc { dst, src, .. } => {
            if !defined(state, src) {
                return false;
            }
            set_kind(state, src, Kind::Num);
            set_kind(state, dst, Kind::Num);
        }
        Op::Bin { dst, lhs, rhs, .. } => {
            if !defined(state, lhs) || !defined(state, rhs) {
                return false;
            }
            set_kind(state, dst, Kind::Num);
        }
        Op::Cmp { dst, lhs, rhs, .. } => {
            if !defined(state, lhs) || !defined(state, rhs) {
                return false;
            }
            set_kind(state, dst, Kind::Bool);
        }
        Op::JumpIf { value, .. } => {
            if !defined(state, value) {
                return false;
            }
        }
        Op::JumpRel { lhs, rhs, .. } | Op::JumpNe { lhs, rhs, .. } => {
            if !defined(state, lhs) || !defined(state, rhs) {
                return false;
            }
        }
        Op::SetAcc { src } => {
            if !defined(state, src) {
                return false;
            }
            state.acc = kind_of(state, src);
        }
        Op::Return => {
            if state.acc != Kind::Num {
                return false;
            }
        }
        Op::Math {
            dst, args, argc, ..
        } => {
            if !args[..usize::from(argc)]
                .iter()
                .all(|&arg| defined(state, arg))
            {
                return false;
            }
            set_kind(state, dst, Kind::Num);
        }
        Op::Jump { .. } | Op::CheckReturn | Op::LoopIter | Op::Nop => {}
    }
    true
}

fn defined(state: &State, reg: u32) -> bool {
    kind_of(state, reg) != Kind::Undef
}

fn kind_of(state: &State, reg: u32) -> Kind {
    state.regs[reg as usize]
}

fn set_kind(state: &mut State, reg: u32, kind: Kind) {
    state.regs[reg as usize] = kind;
}
