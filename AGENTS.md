# Working rules

It is the contract, not a suggestion.

Read [doc/agent-notes.md](doc/agent-notes.md) before starting: it holds what earlier
agents learned that neither these rules nor the [README](README.md) and
[doc/format.md](doc/format.md) state. When a task teaches you something the next
agent would otherwise rediscover, add it there in the same PR.

## Comments

- A comment block is **at most 5 lines**. If the explanation needs more, the code
  is wrong or the explanation belongs in the README or `doc/`.
- One line is the default. Comment *why*, never *what* — the code says what.
- No narrative, no history ("this used to…", "the plan proposed…"), no restating
  the diff, no essays on design philosophy. Git log and `doc/` hold those.
- No dates, commit or PR references, or plan/phase labels in comments. Multi-step
  work is tracked in a working doc under `doc/` or an item in `doc/todo.md`, never
  in code.
- No comment on a self-evident function. `// Lookup returns the address` above
  `func Lookup` is noise.
- Package doc blocks: 5 lines. Files do not get their own prologue.

## Files

- Do not create a file for one or a few helpers. Put them next to what uses them,
  or, if shared, in the package's main or type file.
- A new file needs a new concept, not a new function.
- Prefer editing an existing file over adding one.
- When you edit a file, bring its comments up to these rules; older files predate
  them.

## Scope

- Implement what was asked. Do not add tables, indirection, telemetry, tests or
  abstraction that nothing asked for.
  Exception: a task that explicitly invites additions gets the obvious ones, each
  named in the PR body; ask before any that is not obvious.
- One mechanism per problem. Two mechanisms doing one job is a bug or a refactor
  opportunity.
- Delete before adding. If a change is net-positive lines for a fix, justify it.
- Reverting an existing API to "improve" it is not a fix. Leave working code alone.
- Per-kind data lives in one table indexed by its kind; a new kind adds a row,
  never a switch or a parallel field.
- The module imports only the standard library, and it runs models, never trains
  them: a trainer is anything that writes [doc/format.md](doc/format.md).
- The module is at v0: a change may break the API. The weight file is a contract:
  a change to its bytes bumps `fileVersion` and `doc/format.md` in one commit,
  and `Load` keeps reading the versions trained files carry, since each costs a
  training run to remake. A `Config` field added reads zero for the behaviour
  before it.

## Tests

- One test per rule, named for the rule. No test that restates another.
- Test comments follow the 5-line limit.
- Do not pin lists that a human has to hand-maintain unless the pin prevents a
  real regression.
- Collapse tests: when a broader test covers a narrower one, delete the narrower
  one or do not add it. No test for simple behaviour that is unlikely to fail or
  is already exercised elsewhere.
- `TestWeightFileRoundTrips` is the file's round-trip check: a change to the
  file extends it, never adds a second.

## Docs and commits

- Docs are short. Condense when you touch them; append only for a new concept or
  scope.
- Docs are read in terminals, where a Markdown table renders as its source. Lay
  tabular content out as a nested list; a matrix of figures goes in a fenced code
  block with its columns aligned by spaces, which the terminal keeps. A table that
  neither fits, or that converting would only lengthen, stays a table.
- A gap you are deferring goes in `doc/todo.md`: extend the item it belongs to, or
  add a concise one in the file's pattern.
- PR bodies: what changed, why, how it was verified.
- A change made for performance reports its measurement, before and after, in the
  PR body and in `doc/todo.md`'s speed section: `BenchmarkForward`, or another
  benchmark at a named shape, and the benefit in the unit the user feels (frames
  a second, bytes, allocations). A projection is not a result; a change whose
  measured benefit is nil says so.
- Commits are versions: each commit on main starts with the version it makes
  (`v0.1.1 weights: ...`), the next minor for a break or a new feature and the
  next patch otherwise, and the user tags it. A PR's title is its squashed
  commit's message, so it starts with the version too.
- One task, one branch, one PR. Divide the work into commits on that branch; do not
  split it across stacked PRs or multiple branches.
- Work that continues after its PR merged starts again from the latest main; never
  stack new commits on merged history.
- Finish and push one commit before starting the next.
- Once a PR is merged or closed unmerged, delete its branch on GitHub and locally
  if this session can. A session that cannot, because it has no remote access or
  works in a copy, says so once in its report and leaves the branch; it never
  retries.

## Agents

- Spin off few agents, and only for work one context cannot hold. The account
  quota may run out, which stops every agent in flight at once.
- Agents run in parallel only on finely defined scopes that cannot conflict in
  commits: each brief names the files its agent owns, disjoint from every other
  agent's, and the agent touches nothing else. Shared docs (the README,
  `doc/format.md`, `doc/todo.md`) stay with the parent, which commits each agent's
  work in turn on the task's one branch.
- An agent writes what it finds to a file as it goes, one section per step, under
  the scratchpad or `doc/`, so an interrupt costs a step and not the task; the
  parent reads those files rather than waiting for a final report, and resumes
  from them instead of running the agent again.

## Gates

`go build ./...`, `gofmt -l` on changed files, and `go test ./...`.
- A mechanical edit across files is checked by a count before and after, with a
  search that reads its pattern as the file holds it (`grep -P` for a tab, or a
  literal tab); a search that matches nothing proves nothing, and the PR body
  states the count.
- A change that must not move the outputs (a refactor, a speed-up in the same
  order of sums) leaves a saved file's `Forward` outputs bit-identical before
  and after, on one platform; one that moves them within rounding (a vector
  sum), and any change to the forward pass or the file, runs the float32 check
  against a trainer ([agent notes](doc/agent-notes.md)) and states its reading.
- Do not run `-race` tests unless investigating a known/suspected race issue; CI
  runs them, with the `GOARCH=386`, `GOOS=js GOARCH=wasm` and `GOARCH=arm64`
  builds.
