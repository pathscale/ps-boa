//! Baseline compiler for numeric Boa functions.
//!
//! [`compile_numeric_leaf`] lowers one function body, given as [`ir::Insn`]s, to
//! Cranelift IR in one forward pass and returns machine code owned by a thread-local module. The pointer stays valid until
//! the calling thread exits.

#![cfg_attr(not(test), forbid(clippy::unwrap_used))]

pub mod ir;
mod lower;
mod num;

use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, UserFuncName, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use std::cell::RefCell;

/// Machine code for a numeric leaf.
///
/// `args` points at one f64 per formal parameter, in order.
pub type NumericLeafFn = unsafe extern "C" fn(*const f64) -> f64;

/// Lower `insns`, the engine's translation of one function body.
///
/// `register_count` is the register file size. `parameter_count` is the number
/// of f64 arguments the compiled function reads. Returns `None` when the
/// function is not a numeric leaf or the host cannot emit code.
#[must_use]
pub fn compile_numeric_leaf(
    insns: Vec<ir::Insn>,
    register_count: u32,
    parameter_count: u32,
) -> Option<NumericLeafFn> {
    let prepared = lower::prepare(insns, register_count, parameter_count)?;
    with_jit(|state| state.compile(&prepared)).flatten()
}

enum Cache {
    Uninit,
    Unavailable,
    Ready(Box<JitState>),
}

struct JitState {
    module: JITModule,
    ctx: Context,
    builder_ctx: FunctionBuilderContext,
    rem: FuncId,
    to_int32: FuncId,
    next_id: u32,
    pointer_type: cranelift_codegen::ir::Type,
}

thread_local! {
    static JIT: RefCell<Cache> = const { RefCell::new(Cache::Uninit) };
}

fn with_jit<T>(body: impl FnOnce(&mut JitState) -> T) -> Option<T> {
    JIT.with(|cell| {
        let mut cache = cell.try_borrow_mut().ok()?;
        if matches!(*cache, Cache::Unavailable) {
            return None;
        }
        if matches!(*cache, Cache::Uninit) {
            *cache = match JitState::new() {
                Some(state) => Cache::Ready(Box::new(state)),
                None => Cache::Unavailable,
            };
        }
        match &mut *cache {
            Cache::Ready(state) => Some(body(state)),
            Cache::Unavailable | Cache::Uninit => None,
        }
    })
}

impl JitState {
    fn new() -> Option<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").ok()?;
        flag_builder.set("is_pic", "false").ok()?;
        flag_builder
            .set("enable_nan_canonicalization", "false")
            .ok()?;
        flag_builder.set("opt_level", "speed").ok()?;
        let isa_builder = cranelift_native::builder().ok()?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .ok()?;
        let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
        jit_builder.symbol("boa_jit_rem", num::rem_addr());
        jit_builder.symbol("boa_jit_to_int32", num::to_int32_addr());
        let mut module = JITModule::new(jit_builder);
        let pointer_type = module.target_config().pointer_type();
        let rem = declare_import(
            &mut module,
            "boa_jit_rem",
            &[types::F64, types::F64],
            types::F64,
        )?;
        let to_int32 = declare_import(&mut module, "boa_jit_to_int32", &[types::F64], types::I32)?;
        let ctx = module.make_context();
        Some(Self {
            module,
            ctx,
            builder_ctx: FunctionBuilderContext::new(),
            rem,
            to_int32,
            next_id: 0,
            pointer_type,
        })
    }

    fn compile(&mut self, prepared: &lower::Prepared) -> Option<NumericLeafFn> {
        let mut signature = self.module.make_signature();
        signature.params.push(AbiParam::new(self.pointer_type));
        signature.returns.push(AbiParam::new(types::F64));
        let name = format!("boa_jit_{}", self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        let id = self
            .module
            .declare_function(&name, Linkage::Local, &signature)
            .ok()?;
        self.ctx.func.signature = signature;
        self.ctx.func.name = UserFuncName::user(0, id.as_u32());
        {
            let mut builder =
                cranelift_frontend::FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
            lower::emit(
                &mut builder,
                &mut self.module,
                self.rem,
                self.to_int32,
                self.pointer_type,
                prepared,
            );
            builder.seal_all_blocks();
            builder.finalize(self.module.target_config());
        }
        let defined = self.module.define_function(id, &mut self.ctx);
        self.module.clear_context(&mut self.ctx);
        defined.ok()?;
        self.module.finalize_definitions().ok()?;
        let ptr = self.module.get_finalized_function(id);
        // SAFETY: `ptr` is finalized code for `NumericLeafFn`. This thread's
        // `JITModule` owns the bytes until the thread exits.
        Some(unsafe { std::mem::transmute::<*const u8, NumericLeafFn>(ptr) })
    }
}

fn declare_import(
    module: &mut JITModule,
    name: &str,
    params: &[cranelift_codegen::ir::Type],
    ret: cranelift_codegen::ir::Type,
) -> Option<FuncId> {
    let mut signature = module.make_signature();
    for param in params {
        signature.params.push(AbiParam::new(*param));
    }
    signature.returns.push(AbiParam::new(ret));
    module
        .declare_function(name, Linkage::Import, &signature)
        .ok()
}
