---
name: staged-spec-review
description: Review one stage of a multi-stage specification against the full plan to catch forward-compatibility gaps before they compound. Use whenever the user asks for a review of a single phase, prompt, RFC section, migration step, or implementation that is part of a larger sequence — especially when they provide both the artifact under review AND the full multi-stage spec. Triggers on phrases like "review prompt N against the full spec", "check this phase against the others", "did stage 1 set up correctly for stages 2-5", "is this RFC consistent with the rest", or any request to audit a single piece of a sequenced plan with the rest of the plan as context. Use this even when the user doesn't explicitly say "staged" — if the structure is clearly sequential (prompt 1 of 5, phase A of D, migration step 2, etc.) and they're asking about one piece, this skill applies.
---

# Staged Spec Review

A review pattern for any specification or implementation that is one piece of a multi-stage sequence, where later stages depend on commitments made by earlier ones.

## When this applies

Common situations:
- Multi-prompt LLM workflows (build prompt N of M, review N against the full set)
- Database migrations split across releases
- API versioning where v2 must accommodate v3's planned features
- RFC reviews where one section affects later sections
- Phased product specs where phase 1 sets up phase 4
- Code reviews of one module in a planned multi-module architecture

The unifying property: **the artifact under review makes commitments that downstream stages depend on**, and those commitments are expensive or impossible to change later.

## The core mental model

Reviews of staged work fail in two characteristic ways:

1. **Builder bias.** The person (or model) who wrote the artifact has cognitive investment in their choices. They're inclined to defend rather than question.
2. **Tunnel vision.** Reviews done in isolation check "does this work?" but miss "does this compose with what's coming?"

This skill addresses both: the reviewer is fresh (no investment in the implementation choices) and informed (sees the full sequence, not just the piece under review).

## Required inputs

Before starting a review, confirm you have:

1. **The artifact under review.** The actual prompt, code, spec, or document being audited.
2. **The full multi-stage specification.** All other stages, in their final form, so you can check forward-compatibility.
3. **Stage position.** Which stage this is (e.g., "prompt 1 of 5"). The earlier the stage, the more downstream commitments matter.

If any are missing, ask for them before starting. Reviewing stage N without seeing stages N+1 through end defeats the purpose.

## What to focus on

**Primary focus — forward-compatibility checks.** Walk through each downstream stage and ask:

- Does this stage produce the data structures the next stages will read?
- Do function signatures, schemas, or interfaces match what later stages call?
- Are configuration fields, file paths, or naming conventions used in later stages defined here correctly?
- Are behaviors promised to later stages actually implemented (not just placeholders)?
- Are extension points (optional fields, plugin slots, version markers) in place where later stages will need them?

**Secondary focus — internal correctness.** Standard review concerns: bugs, unclear specs, missing edge cases, untested paths. But these are second priority — a stage that's locally perfect but doesn't compose with stage 3 is a worse outcome than a stage with minor bugs that integrate cleanly.

**Out of scope — design re-litigation.** If the design is committed across the spec, don't propose alternatives. "I would have used SQLite instead of JSON" is noise when the spec is already settled. Flag only when a design choice in this stage *contradicts* a choice in a later stage.

## What NOT to flag

- **Intentionally deferred work.** Placeholder commands, empty directories, stub functions for later stages are expected. Flag only missing infrastructure that *later stages will fail without*, not missing implementations of later stages themselves.
- **Style preferences.** "I'd name this differently" doesn't help unless the name conflicts with later usage.
- **Redundant safety.** If the spec has belt-and-suspenders (e.g., both a programmatic check and an LLM check), don't suggest removing one unless they actively conflict.
- **Anything the spec already explicitly addresses.** Read the full spec before flagging gaps — the answer may be in a later stage.

## Output structure

Always produce three buckets, in this order. This lets the user triage instead of treating every comment as equally urgent.

### 1. Blocking issues
Things that will cause later stages to fail or require rework. The user should fix these before proceeding.

Each item should state:
- What's wrong.
- Which later stage(s) it affects.
- The minimal fix.

### 2. Likely problems
Things that probably cause issues at runtime but won't necessarily block the next stage's implementation. May surface as bugs during integration testing or in production edge cases.

Each item should state:
- What might go wrong.
- Under what conditions.
- Suggested mitigation (which may be "document the limitation" rather than "fix the code").

### 3. Nitpicks
Style, optional improvements, "while you're in here" suggestions. The user can ignore these without consequence.

Keep this section short. If it's longer than the first two combined, you're padding.

## Tactical guidance

**Read the full spec before reviewing the artifact.** Sounds obvious, but the temptation is to start commenting as you read the artifact and only later check downstream. Resist it. A complete mental model of the full spec lets you catch issues that scattered reading misses.

**Trace specific data flows.** Pick a key data structure (a schema, a config field, a file path) and trace it from where this stage defines it through every later stage that uses it. Mismatches surface fast this way.

**Trace specific failure modes.** Pick a failure scenario (crash mid-stage, malformed input, missing dependency) and walk through how this stage handles it AND how later stages would react to that handling.

**Be terse.** A blocking issue needs enough detail to act on, not a paragraph of context the user already has. The user wrote the spec; they don't need it re-explained.

**Avoid hedging language.** "Maybe consider possibly looking at..." wastes the user's attention. If you're not confident enough to say it directly, don't say it.

**State your reasoning when it might not be obvious.** If you flag something as blocking, briefly say why later stage X needs it. The user can then verify your reading or push back if you're wrong.

## What good reviews look like

A good review of a 5-stage spec's first stage might produce:

- 2-4 blocking issues (forward-compatibility gaps that will bite at stage 2 or 3)
- 3-6 likely problems (edge cases, ambiguities, runtime concerns)
- 1-3 nitpicks

If you're producing 15 blocking issues, you're either reviewing a genuinely broken spec or you're miscategorizing. Re-read your blocking section and demote items that won't actually block.

If you're producing zero blocking issues on a substantial spec, you're probably missing something. Re-read with specific data flows in mind.

## What bad reviews look like

- Long preamble before the actual findings.
- Restating what the spec says before commenting on it.
- "Pros and cons" framing on already-decided choices.
- Mixing review with "here's what I would have done differently."
- Treating optional improvements as blocking issues.
- Comments that boil down to "this could be better" without specifying how.

## Tone

Be direct. The user explicitly asked for review precisely because they want to find problems before they compound. Softening blockers as suggestions or burying them in praise wastes the opportunity. They will not be offended by clear "this is wrong, here's why" — they'll be relieved you caught it.

But: don't manufacture problems to seem thorough. If a stage is genuinely solid, say so and keep the review short. A two-issue review of a clean spec is a better outcome than a fifteen-item review padded with nitpicks.
