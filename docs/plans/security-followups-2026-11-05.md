# Security findings to revalidate: ps-boa

This queue carries forward candidate findings from a static review dated 2026-10-14. That review did not run the software. None of these findings has been revalidated against the current source in this checkout in this cleanup, so treat every item as unknown, not as a confirmed current defect.

For each item, compare the reported location with current source and workflow configuration. If it remains open, record the concrete owner task and expected focused evidence in the project TODO. If fixed or refuted, record the source evidence and remove the item. Do not copy historical exploit examples or private machine paths into public docs.

## Candidate findings

- [ ] Revalidate H1: High: the parser and bytecompiler have no nesting depth limit (reported location: core/parser/src (whole crate). No depth counter or stacker exists anywhere in core/parser/src, core/ast/src or core/engine/src.). Current status: unknown.
- [ ] Revalidate H2: High: native-to-native recursion is unbounded and aborts the host (reported location: core/engine/src/object/operations.rs:447 (JsObject::call native shortcut), core/engine/src/builtins/json/mod.rs:877/1069/1218 (stringify recursion), core/engine/src/builtins/array/mod.rs:1925 (flatten_into_array), and Array join/toString.). Current status: unknown.
- [ ] Revalidate H3: High (raised from medium): JSON.stringify preallocates an attacker-chosen length (reported location: core/engine/src/builtins/json/mod.rs:1247). Current status: unknown.
- [ ] Revalidate M1: Medium: the set inline cache writes through frozen or non-writable properties on unique-shape objects (reported location: core/engine/src/vm/opcode/set/property.rs:18-39 (cached_set_own), :108 (set_by_name hit path), core/engine/src/object/shape/unique_shape.rs:150-159). Current status: unknown.
- [ ] Revalidate M2: Medium: the execution time limit is not enforced inside native work (reported location: core/engine/src/vm/execution_budget.rs:81, core/engine/src/builtins/regexp/mod.rs:1240-1280 and 2176-2244, the array builtins). Current status: unknown.
- [ ] Revalidate M3: Medium (lowered from high): numeric-leaf JIT code ignores the deadline, the interrupt flag and the loop iteration limit (reported location: core/jit/src/lower.rs:253 and 763, core/engine/src/vm/jit.rs:284 and 149). Current status: unknown.
- [ ] Revalidate M4: Medium (lowered from high): the crates.io token is in the environment of unpinned dependency build scripts (reported location: .github/workflows/publish.yml:80 and 110. .gitignore:61 drops Cargo.lock.). Current status: unknown.
- [ ] Revalidate M5: Medium: the secret-bearing publish job uses mutable action refs (reported location: .github/workflows/publish.yml:46, 50, 52 (actions/checkout@v7, dtolnay/rust-toolchain@stable, Swatinem/rust-cache@v2)). Current status: unknown.
- [ ] Revalidate M6: Medium: the upstream release.yml is still live and is a second publish path that holds the token (reported location: .github/workflows/release.yml:3-4, 32-49, 79). Current status: unknown.
- [ ] Revalidate L1: Low (lowered from medium): publish can be dispatched from any branch (reported location: .github/workflows/publish.yml:22). Current status: unknown.
- [ ] Revalidate L2: Low (lowered from medium): JIT code memory is never freed (reported location: core/jit/src/lib.rs:182-257, thread_local JIT at around line 262). Current status: unknown.
- [ ] Revalidate L3: Low: baseline JIT frames do not yield inside run_async_with_budget (reported location: core/engine/src/vm/mod.rs:1127, core/engine/src/vm/jit.rs:396-415). Current status: unknown.
- [ ] Revalidate L4: Low: the thread-local regex cache retains attacker patterns after the Context is dropped (reported location: core/engine/src/builtins/regexp/mod.rs (MATCHERS thread_local, cap of 1024 entries)). Current status: unknown.
- [ ] Revalidate L5: Low: the nightly job ships lockfile-less binaries with contents: write (reported location: .github/workflows/nightly_build.yml:14, 47-56). Current status: unknown.
- [ ] Revalidate I1: Info: the bytecode verifier only exists in debug builds (reported location: core/engine/src/vm/mod.rs:56-57). Current status: unknown.

## Review limit

This note preserves issue titles and the source locations reported at review time. It is not a source audit, fix plan, security certification, or publication clearance.
