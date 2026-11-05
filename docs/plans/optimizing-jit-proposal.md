# Optimizing JIT proposal

Status: proposal only. This document records a possible future direction; it does not
authorize implementation or change the fork's current scope. Preserve ps-boa as a thin
Boa fork. Prefer upstream fixes, keep crate and Rust library names intact, and isolate
any accepted fork-specific work in a small, reviewable diff.

## Intended architecture

Keep the interpreter complete and authoritative, with one optional optimizing tier
above it. Keep the existing numeric JIT as a comparison point until an alternative
has demonstrated correctness and benefit. New compiled execution must be opt-in until
its semantic, safety, and performance gates are accepted. Do not add another
helper-per-bytecode baseline or a separate quickening interpreter: previous supplied
measurements found those approaches unhelpful for this engine.

The candidate compiler would build bounded control-flow regions from Boa bytecode,
construct explicit SSA with JavaScript representations and effects, then lower through
a small machine IR. Prefer direct bytecode-to-IR construction over a trace recorder,
whole-program inference, or a general-purpose optimizer framework. Start with local
constant propagation, value forwarding, numeric narrowing, and dominated-check
elimination only where JavaScript semantics, aliasing, and effect ordering are
explicitly preserved.

Compilation admission should account for expected useful work, compile cost, exit
frequency, failed imports, and code pressure. Record already-observed feedback with
cheap stores; avoid additional interpreter branches in hot handlers. Keep property
case capacity distinct from the number of cases emitted in native code. Bound region
size, live values, control-flow edges, frame states, inline expansion, OSR variants,
and total code/metadata so one synchronous compile cannot consume unbounded work.

## Object access and borrow safety

Compiled property access needs authoritative storage shared with the interpreter,
not a mirror refreshed after each interpreter mutation. A fast access must validate
the applicable object category, shape/layout facts, element kind, index bounds, and
any result-type assumption before directly loading or storing a slot. Begin with
own-properties and dense elements; inherited properties, mutable unique-shape
assumptions, and holes require separately proven dependencies.

Any raw access proposal must preserve Rust aliasing and Boa's borrow semantics. If
borrow tracking is limited to JIT-eligible objects, promotion must happen only after
existing borrows are quiescent or through a mechanism that accounts for every
preexisting guard. Defer promotion where that guarantee cannot be established.
Do not add bookkeeping to every interpreter mutation unless a measured, reviewed
design proves it necessary.

## Exits, calls, and garbage collection

First evaluate one-frame exits that return to the interpreter's existing owning
register homes. Add supplemental roots only for live values without an owning home.
Keep exits available at instruction boundaries; reconstruct state only when values
are unboxed, frames are inlined, or optimized objects need materialization. Frame
states must cover exception edges, deoptimization, and materialization, and must not
replay effects already made observable.

Define three related but distinct metadata contracts: GC root locations, interpreter
resume/reconstruction states, and native unwind/frame-walking information. Generate
them from consistent machine-location assignments, but do not treat one as a
substitute for another. Explicitly cover pointers held in registers and frames at
safepoints, live embedded heap references, code pinning, invalidation, and return
continuations when a caller is invalidated during a callee or native callback.

Calls and loop backedges need independent admission so a function called once can
still optimize a long-running loop. Preserve Boa's actual frame and operand-stack
state during OSR. Compiled-to-compiled calls and direct native wrappers are later
steps; they must preserve closure identity, environments, arguments, realms,
exceptions, reentry, and logical call/stack limits. Bound inlining by reconstruction
cost, live values, root spills, frame depth, and recipe size as well as source size.

## Backend and executable memory

If a custom backend is pursued, keep semantic IR separate from machine IR and begin
with a narrow target/instruction set and a simple allocator. Validate fixed-register
constraints, call clobbers, phi moves, spills, root/deopt uses, branch ranges, and
publication before code becomes executable. Any in-place patching needs an explicit
safe write and publication protocol. Embedded references must remain traced and
valid for the lifetime of code that uses them.

The possible shared backend with another project remains an external owner decision. This
proposal does not select a shared crate, add a dependency, or require an AOT mode.
The prior AOT direction was explicitly removed from the JIT scope; do not reintroduce
it as a prerequisite.

## Adoption sequence and stop conditions

1. Establish a reproducible baseline and instrumentation plan using existing,
   owner-approved workloads. Keep corpus and result contents outside this plan.
2. Prove authoritative storage, eligibility promotion, and retained-borrow exclusion
   before expanding compiler scope.
3. Prove root publication and cheap one-frame exits, including nested/native reentry
   and forced collection, before broadening optimized execution.
4. Add bounded SSA construction, explicit effects, frame states, and phase-work caps.
5. Add a minimal native path and loop OSR; then consider property access, stores,
   mutation dependencies, compiled calls, and bounded inlining in that order.
6. Consider allocation sinking only after identity, aliasing, escape, and exit
   materialization are correct. Add another architecture only after the first target
   has a stable contract.

Each stage needs an agreed correctness gate, overhead/regression limit, memory/code
budget, and retirement rule before implementation starts. Attribute results to
compiled coverage, helper work, exits, compilation, allocation, and collection rather
than relying on one aggregate timing. If an early storage or root-safety gate fails,
stop and retain the interpreter; do not add a new tier to route around it. Do not add
automatic CI workflows for this proposal; any future workflow must be manual by
default and separately reviewed.

## Evidence and limits

This proposal consolidates the JIT plan, folded design, review lessons, speed plan,
and three architecture surveys supplied on 2026-10-12. Their supplied Boa
measurements were not independently reproduced for this document. Cross-runtime
examples support design questions and tradeoffs, not predicted performance for Boa.
The exact implementation details, gates, current code locations, workloads, and
shared-backend decision must be revalidated against the then-current upstream base
and owner requirements before any work begins.
