---
title: The Three Feedback Loops of AI-Assisted Development
source: Andrew Ng, The Batch, Issue 359 (deeplearning.ai)
tags: [ai-engineering, agentic-coding, mental-model, spec-driven-development]
status: reference
audience: human-and-agent
---

# The Three Feedback Loops of AI-Assisted Development

> Origin note: this framework was articulated by Andrew Ng in The Batch, Issue 359, in the context of building AI-powered products with coding agents. This document expands it into a durable mental model — for a human deciding how to spend their attention, and for an AI agent (like Claude) that ingests this note and needs to know what role it's playing at any given moment.

## Why this document exists

Modern AI-assisted development doesn't have one feedback loop — it has three, nested inside each other, running at different speeds. Most confusion about "how much should I let the agent do on its own" or "what am I actually for, if the agent writes the code" comes from not distinguishing which loop you're in. This document names the three loops, explains the mechanics of each, and gives concrete signals for recognizing which one you're operating in right now.

This is not tied to any specific project. It's a lens for *any* work where an agent writes/tests/iterates and a human is in the loop somewhere. Read it once for the concepts; return to it when you're unsure whether you're micromanaging a loop that should run itself, or neglecting a loop that needs your judgment.

---

## The core idea in one paragraph

Building something with an AI agent is not "the agent codes, I review." It's three concentric loops running at three different timescales — minutes, hours, days — each with a different owner, a different kind of feedback, and a different failure mode when it's skipped or done by the wrong party. Get the loop assignment wrong (e.g., a human manually re-checking things an agent's test loop should catch, or an agent trying to make product-taste calls no test suite can validate) and you waste the whole system's leverage.

---

## Loop 1: The Agentic Coding Loop (minutes)

**Who drives it:** The agent, autonomously.
**Timescale:** Minutes to (with a good spec) hours, unsupervised.
**Mechanic:** Write code → run tests → observe failure → revise → repeat, without a human in the middle of each cycle.

### What makes this loop work
This loop only runs well when the *spec is verifiable*. "Fix the bug" is not a spec an agent can iterate against — there's no test that tells it when it's done. "Given input X, output should be Y; here is a failing test that reproduces the current behavior" is a spec an agent can loop against unsupervised, because the test *is* the feedback signal. This is precisely the discipline underneath "turn tasks into verifiable success criteria" — write the failing test first, then let the loop close itself.

The tighter and more automatic the feedback (compiler errors, unit tests, type checkers, linters, a runnable app that either starts or doesn't), the longer this loop can run without a human touching it. The looser the feedback (vibes, "does this look right"), the faster it needs to escalate to Loop 2.

### Failure modes
- **Underspecified goals**: the agent has no way to know it's done, so it either stops too early (declares success without verification) or loops forever making cosmetic changes.
- **No ground truth**: if "correct" can only be judged by a human eyeballing it, this isn't a Loop 1 task — it's mislabeled Loop 2 work.
- **Reward hacking against the test**: an agent can satisfy a badly written test without satisfying the actual intent (e.g., hardcoding the expected output). A verifiable spec is necessary but not sufficient — the test itself has to be honest.

### Signal that you're in this loop
You're waiting on a test run, a build, or an agent's self-correction cycle. Your job right now is to have written a good spec *before* this started — not to intervene mid-loop unless it's stuck.

---

## Loop 2: The Developer Feedback Loop (hours)

**Who drives it:** The human, reviewing.
**Timescale:** Hours — a working session, a day.
**Mechanic:** Look at what the agent produced (a running feature, a UI, a PR) and decide what happens next.

### The key reframe
Ng's central point here is a role change, not just a cadence change: once agents are competent at finding *bugs* (Loop 1 handles correctness), the human's value in Loop 2 stops being "find what's broken" and becomes **"decide what matters."** That's:
- Feature prioritization — what should get built next, and what shouldn't
- UX and product judgment — does this *feel* right, not just "does it pass"
- Scope and taste calls — is this the right abstraction, the right amount of polish, the right tradeoff

This is a genuinely different skill from code review. It requires *context the AI doesn't have* — knowledge of the user, the business, the point of the product — which is why Ng calls it a "context advantage." An agent can tell you the code works. Only a human (for now) can tell you the code works *on the right thing*.

### Failure modes
- **Reverting to QA habits**: spending Loop 2 time re-deriving bugs a test suite should have caught. This is wasted human attention — push that verification back into Loop 1's spec instead of doing it by hand every cycle.
- **Reviewing too late or too rarely**: if Loop 2 doesn't run for days, Loop 1 drifts — the agent optimizes hard against a spec that was already stale.
- **Reviewing too eagerly**: interrupting a Loop 1 run that's still legitimately iterating, before it's had a chance to converge.

### Signal that you're in this loop
You're looking at a working artifact and asking "is this the right thing," not "does this technically work." If you catch yourself manually retracing test coverage, that's a sign the spec in Loop 1 needs to be tightened, not that you personally need to check harder.

---

## Loop 3: The External Feedback Loop (days)

**Who drives it:** Users, alpha testers, real-world signal — mediated by the human/team, not the agent.
**Timescale:** Days to weeks.
**Mechanic:** Ship something real (even narrowly) → collect actual usage feedback, A/B results, or alpha-tester reactions → let that reshape the *product vision*, which then reshapes what gets fed into Loop 2 and Loop 1.

### What this loop is actually for
Loops 1 and 2 can make something correct and well-crafted that nobody wants, or that solves the wrong problem. Loop 3 is the check against building the wrong thing well. It's slow by nature — you can't compress "does a real user find this valuable" into a unit test or a single afternoon of review — and that's fine, because it's not supposed to run at the same cadence as the other two.

Ng's observation that engineers are increasingly playing an expanded product-management role lives here: someone has to *own* the tension between "keep building" and "go find out if this is even right," because nothing in Loop 1 or Loop 2 will surface that tension on its own.

### Failure modes
- **Never closing it**: iterating forever in Loops 1–2 without ever exposing the work to real signal — polishing something whose fundamental direction was never validated.
- **Over-indexing on it too early**: seeking external feedback on something so unfinished that the feedback is noise (people can't evaluate a broken prototype meaningfully).
- **Not feeding it back**: collecting feedback but not translating it into revised specs for Loop 1 or revised priorities for Loop 2 — external feedback that dies in a doc nobody rereads.

### Signal that you're in this loop
You're asking "should this exist at all, in this form" rather than "does this work" or "is this well built." The answer here changes what you tell Loop 2 to prioritize, which changes the specs you write for Loop 1.

---

## How the three loops nest

```
Loop 3 (days):    [ vision / direction ]
                        |
                        v  (shapes priorities)
Loop 2 (hours):   [ human judgment / prioritization / taste ]
                        |
                        v  (produces a verifiable spec)
Loop 1 (minutes): [ agent writes -> tests -> iterates -> converges ]
                        |
                        ^  (produces a working artifact)
                        |
                  feeds back up into Loop 2 for review,
                  and eventually Loop 3 for real-world signal
```

Each loop's *output* is the next loop's *input*. A good spec in Loop 1 comes from a clear priority decision in Loop 2. A clear priority decision in Loop 2 comes from real signal surfaced in Loop 3. When something feels broken in this system, the fix is almost always "push this decision to the correct loop," not "work harder in the loop you're already in."

---

## Practical checklist (for a human using this)

Before starting agentic work, ask:
1. **Is there a verifiable spec, or do I need to write one first?** (Loop 1 gate — no test, no unsupervised loop.)
2. **When the agent's output comes back, am I checking correctness or checking priority/taste?** (Don't do Loop 1's job in Loop 2 — tighten the spec instead.)
3. **Has this been exposed to anything outside my own judgment recently?** (If it's been purely Loop 1/2 for a long stretch, it's time to force a Loop 3 checkpoint, even a small one.)
4. **When feedback comes in from Loop 3, did it actually change a spec or a priority — or did it just get noted and ignored?** (Unclosed loops are wasted signal.)

## Practical checklist (for an agent ingesting this document)

If you (an AI agent) are reading this note as context for a task:
1. **Assume you own Loop 1 by default.** If the task has a verifiable spec (a failing test, a clear input/output contract, a reproducible bug), iterate on it autonomously rather than pausing after every small step to ask for confirmation.
2. **If the task has no verifiable spec, say so before starting** — that's a Loop 2 gap, not something to paper over by guessing at "done."
3. **Do not make Loop 2 or Loop 3 decisions on the human's behalf** — prioritization, product taste, and "is this worth shipping" are explicitly the human's context advantage in this framework, not the agent's job to infer silently.
4. **When surfacing your work for review, distinguish "this is verified correct" from "this is a judgment call I'm flagging"** — collapsing those two into one report defeats the purpose of separating the loops.

---

## Relationship to spec-driven / test-first development

This framework isn't new mechanics — it's a naming of what disciplined test-first development already implies, extended to account for a second worker (the agent) capable of running the inner loop unsupervised. The old adage "write the test first" was always partly about creating a verifiable spec; what's changed is that a competent agent can now consume that spec directly and iterate against it without a human present for every cycle. The bottleneck has moved from "who writes the code" to "who writes the spec, decides what's worth building, and decides when real-world feedback should redirect the work." Loops 2 and 3 are where that bottleneck now lives.

---

## One-line summary

**Loop 1 (minutes) closes itself against a verifiable spec. Loop 2 (hours) is human judgment on priority and taste, not bug-hunting. Loop 3 (days) is reality checking the whole direction. Confusion in this system is almost always a decision running in the wrong loop.**
