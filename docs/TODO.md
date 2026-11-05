# TODO

## Future optimizing JIT proposal

The [optimizing JIT proposal](plans/optimizing-jit-proposal.md) is retained as an
unadopted direction, not an implementation commitment. Keep this fork thin and
upstream-friendly. Before any JIT work is authorized:

- Reconcile the proposal with current Boa/upstream code and the fork's current
  numeric JIT; remove obsolete code-location assumptions.
- Obtain owner decisions on the object-storage/borrow contract and whether any
  backend sharing with another project is wanted. Do not introduce a cross-repository
  dependency without that decision.
- Define correctness, interpreter-overhead, regression, compile-work, and memory
  gates from currently approved workloads; do not copy historical thresholds or
  treat old measurements as current evidence.
- Review GC roots, frame reconstruction, exception/effect ordering, invalidation,
  return continuations, and executable-memory publication before enabling compiled
  paths.
- Keep JIT behavior disabled by default until its gates pass. Preserve the complete
  interpreter as the semantic fallback.
- Keep any proposed GitHub workflow manual by default. No automatic CI workflow is
  authorized by this plan.

No implementation task is currently approved by this TODO entry.

## Security follow-up

- [ ] Revalidate the historical candidate findings in [security follow-ups](plans/security-followups-2026-11-05.md) against current source. This review has not established which findings remain open.
