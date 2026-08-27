# Working agreement: ps-boa

Operating contract for any coding agent here, and the single source of truth. Codex,
Cursor and Gemini read `AGENTS.md` natively; Claude Code imports it from `CLAUDE.md`.
Never fork these rules into a per-vendor file.

## What this repository is

A PathScale fork of [boa-dev/boa](https://github.com/boa-dev/boa), the JavaScript
engine `ps-blitz-script` runs. It exists for one reason: crates.io refuses a crate with
a git dependency, so `ps-blitz-script` could not be published while it took `boa_engine`
from a git revision. The fork is published under `ps-` names so the whole engine stack
can be consumed from the registry.

**Every published crate is renamed, the Rust names are not.** `package.name` is
`ps-boa-engine`; `lib.name` stays `boa_engine`. So `use boa_engine::...` is unchanged in
every consumer and in all 1,453 source references here, and only the manifests differ
from upstream. Keep it that way: renaming the lib names would turn a manifest change
into a rewrite of the whole tree and make every future upstream merge a conflict.

## Remotes

- `origin` is `git@github.com:pathscale/ps-boa.git`. This is where work goes.
- `upstream` is `https://github.com/boa-dev/boa.git`, fetch only. Never push there.

## Staying close to upstream

The fork's value is that it is boring. Carry local changes as a thin layer:

- Do not reformat, rename modules, or restructure. A diff against upstream should be
  readable in one screen per crate.
- Prefer taking a fix from upstream over writing our own.
- When merging upstream, expect conflicts only in `Cargo.toml` files. If a conflict
  reaches a `.rs` file, ask whether the local change should exist at all.

## Publishing

Crates publish bottom-up, because a crate cannot be published until everything it
depends on is on the registry:

    ps-boa-string, ps-boa-macros, ps-boa-icu-provider   (leaves, no internal deps)
    ps-boa-interner, ps-boa-gc
    ps-boa-ast
    ps-boa-parser
    ps-boa-engine
    ps-boa-runtime, ps-boa-wintertc

Dry-run first (`cargo publish -p <crate> --dry-run`), and never publish from a dirty
tree. A published version can never be replaced, only yanked.

## Git workflow

- Work on a branch, ship through a pull request, never commit to `master`.
- **Never squash. Rebase-merge only.** Squashing destroys the per-commit trail.
- **Never create merge commits**, not even to refresh a branch. Rebase onto the moved
  base and force-with-lease.
- No AI attribution anywhere: no `Co-Authored-By`, no "Generated with" footer, in
  commits, pull requests or issues.
- No em dashes in commit messages. Comma, colon, parentheses or a full stop.

## Before delivery

    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

Upstream's own CI is the reference for what "green" means here. A failure that also
fails on `upstream/main` is not ours to fix in a delivery branch; say so and move on.

## Invariants (do not break these)

- **No Python.** Not a script, not `python3 -c`, not a heredoc. Reaching for it is the
  tell that a step is being solved by parsing when the tool that owns the answer could
  just be asked. Do not swap it for another parser either, and do not assume `jq` is
  present: it does not ship with macOS. A fixed-shape field is one `sed -nE` line;
  anything needing real parsing belongs in this repo's own language, where it can be
  tested. If a task seems to need Python, the approach is wrong.
