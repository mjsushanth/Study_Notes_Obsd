---
title: FDE Technical Rundowns and Live Scenarios
type: interview-prep
role: Forward Deployed Engineer
created: 2026-08-12
tags:
  - interview/fde
  - interview/technical
  - career/2026
---

# 02 - Technical Rundowns and Live Scenarios

Companion notes: [[00 - FDE Core Positioning and Screen Prep]] · [[01 - FDE Story Bank - Client Facing Translations]]

The FDE test is not whether you know RAG. It is whether you can explain RAG to a CFO in forty
seconds, to a skeptical staff engineer in five minutes, and pick the right one without being told
which room you are in.

---

## Part 1 - The three-altitude method

Every technical concept gets three prepared versions. Practice switching mid-sentence when someone
new joins the call.

| Altitude | Audience | Length | Contains | Never contains |
|---|---|---|---|---|
| **Board** | CEO, CFO, PE operating partner | 30-45 sec | The business decision it changes, what it costs, what it risks | Any component name, any framework |
| **Bridge** | VP Ops, Head of Sales, a technical PM | 2-3 min | The approach, the main trade-off, why the alternative was rejected | Line-level implementation |
| **Bench** | Their staff engineer, your own team | 5+ min | Architecture, failure modes, measurements, what you would change | Business framing they already have |

**The tell that you have the altitude wrong:** at board level, if your sentence contains a noun the
listener could not define, you have already lost them. At bench level, if you are still explaining
*why* rather than *how* after ninety seconds, they think you are non-technical.

**The transition sentence that buys you time:** *"Do you want the short version or the actual
architecture?"* Perfectly professional, and it makes the client choose the altitude for you.

---

## Part 2 - Rundowns

### RAG (Retrieval-Augmented Generation)

**Board:**
> "The model doesn't know your business, and if you ask it anyway it will make something up
> confidently. So instead of asking it to remember, we let it look things up in your documents
> first and then answer from what it found - with a citation back to the page. The value is that
> someone can check the answer. The cost is that it's only as good as the documents you give it."

**Bridge:**
> "Three stages. We break your documents into chunks and index them so they're searchable by
> meaning rather than keyword. At question time we retrieve the most relevant chunks. Then we hand
> the model the question *and* those chunks and ask it to answer only from them.
>
> The failure mode people underestimate is retrieval, not generation. If the right chunk never gets
> pulled, no model fixes that - it just writes a fluent wrong answer. So most of the engineering
> effort goes into retrieval quality and into making the system say 'I don't have that' instead of
> guessing."

**Bench:**
> "In FinSights it was S3 Vectors for the store, with a pipeline running about 1,850 vectors a
> minute with checkpointing and token/RPM controls. Retrieval used query variants against filtered,
> global, and union regimes, then sentence-window expansion, dedup, and stratified distance
> selection rather than plain top-k, because top-k on a corpus with near-duplicate boilerplate
> gives you five copies of the same paragraph.
>
> I tested cross-encoder reranking and it *degraded* results on this corpus, so it isn't in the
> system. Evaluation was deterministic neighbor tests for retrieval plus BLEURT-20, ROUGE-L,
> BERTScore and LLM judges on generation - 0.82 to 0.86 BERTScore F1 on business-realistic tests.
> About two cents a query, per-stage latency instrumented."

---

### Agents and agentic workflows

**Board:**
> "A regular AI call answers one question. An agent is given a goal and allowed to take several
> steps to get there - look something up, do a calculation, check a system, then decide what to do
> next. It's more capable and it's genuinely less predictable, which is why the engineering is
> mostly about putting boundaries around it: what it's allowed to touch, when it has to stop, and
> what a human has to approve."

**Bridge:**
> "The useful mental model is a state machine, not a chatbot. You define the steps, what can loop,
> what has to happen before what, and where it terminates. The three things that separate a demo
> from something you'd deploy: it can resume if it dies partway through, it can't take an unbounded
> number of actions, and every step is logged well enough that you can reconstruct why it did what
> it did.
>
> That last one matters more with a client than any capability question. When an agent does
> something wrong - and it will - 'we can see exactly which step went wrong' is the difference
> between a fixable incident and losing the account."

**Bench:**
> "My system was a 13-node LangGraph state machine - gates, cycles, map-reduce fan-out, typed
> reducers, SQLite checkpointing. I forced a recursion failure deliberately to verify a partial run
> resumes without duplicating completed turns, because that path never gets tested by a happy-path
> demo.
>
> Tools were exposed through two MCP servers - nine tools, seven resources - behind a shared
> adapter, so the same retrieval capability runs in-process or over the protocol. Measured 15.3 ms
> in-process versus 23.1 ms over MCP, which is a real number to hand someone deciding whether the
> abstraction is worth it. Grounding was hybrid BM25 plus dense with reciprocal rank fusion over
> 14,195 passages."

---

### MCP (Model Context Protocol)

**Board:**
> "It's a standard way to plug AI systems into your existing tools and data - so you write the
> connection to your CRM once instead of once per AI product you adopt."

**Bridge:**
> "Without a standard, every AI integration is bespoke, so you end up with the same connector
> written five different ways with five different security postures. MCP puts a defined boundary
> between the model and your systems - which is also the natural place to put access control and
> auditing, since everything the model can reach has to come through it."

**Bench:**
> "I wrote two servers exposing nine tools and seven resources behind a shared adapter so the same
> capability is callable in-process or over the protocol. That let me measure the actual cost of
> the boundary - 15.3 ms versus 23.1 ms - rather than argue about it. The interesting design
> question is granularity: too fine and the model burns turns chaining calls, too coarse and you've
> just wrapped your whole API in one tool and lost the point."

---

### Evaluation (the thing that makes you sound senior)

**Board:**
> "Before this goes in front of your people, we need to know how often it's right, how it's wrong
> when it's wrong, and what it costs per use. Otherwise we're asking your team to trust something
> nobody has measured."

**Bridge:**
> "Evaluation splits in two, and conflating them is the common mistake. Retrieval evaluation asks
> 'did we find the right source material' - that's testable deterministically. Generation
> evaluation asks 'was the answer good,' which needs a mix of automated scoring and human or
> model-judged review against cases *you* define as realistic. Vendor benchmark numbers tell you
> almost nothing about your documents.
>
> The practical version for an engagement: build a small set of questions your business actually
> asks, with answers you agree are correct, before building the system. That set is the contract."

**Bench:**
> "Deterministic neighbor tests on retrieval, then BLEURT-20, ROUGE-L, BERTScore and LLM judges on
> generation, scored against business-realistic cases rather than the easy ones. That's how I
> caught the reranking regression - the ablation showed a component that's in every reference
> architecture actively hurting this corpus. Without the eval harness I'd have shipped it because
> it's supposed to help."

---

### Entity resolution (your deepest well - see [[01 - FDE Story Bank - Client Facing Translations]] Story 1)

**Board:**
> "After an acquisition, the same customer exists in both companies' systems under slightly
> different names, with no shared ID. Until that's fixed, every combined number is wrong - you'll
> double-count customers, and you can't tell which salesperson earned what. It's unglamorous and
> it's the thing that has to happen before anything else works."

**Bridge:**
> "You match in rounds. Each round tries a different construction of the same entity - name orders
> reversed, middle names present on one side only, preferred versus legal names, an employee ID
> where one exists. You deliberately match loosely and then resolve the duplicates that creates
> with explicit rules, because a strict match silently drops records and you never find out.
>
> Two things I'd insist on with any client: track the unmatched residue as a visible number, and
> build a manual-override path from day one. There is always a set the algorithm won't get, and if
> there's nowhere durable to put the human decision, someone re-fixes the same twenty records every
> month in a spreadsheet."

**Bench:** the full technical version and all anchor numbers are in Story 1.

---

### Data pipelines and warehousing

**Board:**
> "Right now your numbers live in six systems that disagree with each other. The work is landing
> them in one place, on an agreed definition, refreshed on a schedule people can rely on - so that
> when two executives quote a number in the same meeting, it's the same number."

**Bridge:**
> "Land the raw data, transform it into a modeled layer with agreed business definitions, and serve
> a read layer to reporting. Keep those tiers separate so a reporting change can't break ingestion.
> Every load logs what it touched, how many rows, and whether it failed - if you can't answer 'did
> last night's load actually work' in one query, you don't have a platform, you have scripts.
>
> And define the history strategy per subject area rather than globally. Some things you need to
> see as-of the decision date; some things only need current state. Applying one policy everywhere
> is how you get either a bloated warehouse or an unanswerable audit question."

**Bench:**
> "At Innova: Oracle Autonomous Data Warehouse, four-tier schema - staging, curated star, read-only
> reporting, each mirrored for UAT and DEV. 306 schema-qualified objects, 84 fact and 42 dimension
> tables. 39 stored procedures, roughly 30,700 lines. Eight-plus sources including Bullhorn Data
> Mirror on SQL Server, four Oracle EBS instances, JobDiva, Salesforce, and the Coupa API including
> a reverse write-back.
>
> Two pieces I'd call genuinely hard. Column-level CDC replay: rather than trusting a snapshot, the
> job-order fact is rebuilt by replaying the ATS's own audit history row by row - branching on
> column name for target datatype, detecting which of two different text date formats a history row
> used, repairing type drift where IDs are strings in history but numbers in the live table, and
> wrapping each row in an exception handler so one malformed row can't abort the load. And SCD Type
> 2 end-dating on the transactional facts, with FP&A history handled instead as 48 physical monthly
> snapshot tables - two different history strategies chosen deliberately per subject area.
>
> Observability: a shared run-audit table referenced 363 times, with the larger procedures logging
> per stage rather than per run."

---

### Cost and latency (consultants love this and most engineers skip it)

**Board:**
> "Per-use cost matters more than model quality once you're at volume. A two-cent query is fine for
> an analyst asking ten questions a day and completely different if you're putting it in a customer
> support flow at ten thousand a day. I measure it before we scale it, not after the invoice."

**Bridge:**
> "Instrument per stage, not per request. When something gets slow or expensive it's almost always
> one stage - an oversized retrieval, a model call that could be a smaller model, a redundant
> embedding pass. Aggregate numbers tell you there's a problem; per-stage tells you where."

**Bench:** ~$0.02/query on FinSights with live cost tracking in the embedding pipeline, token and
RPM controls, per-stage latency inspection, and phase-separated model residency in the local system
to serve ~34 GB of weights on 24 GB of memory.

---

### When NOT to use AI (the most senior thing you can say)

Have this ready. A consultancy that sells AI needs people who will not build the wrong thing,
because a failed engagement costs them the account.

> "A lot of what gets scoped as an AI problem is a data problem wearing a costume. If the client
> can't get a consistent customer list out of their systems, an agent on top of that inconsistency
> just produces confident wrong answers faster. The honest sequence is usually: fix identity and
> definitions first, then automate.
>
> And if the task is deterministic and high-stakes - commission calculations, revenue recognition,
> anything auditable - a rules engine you can read is better than a model you have to evaluate.
> I'd rather use the model where the input is genuinely unstructured and the output gets reviewed."

---

## Part 3 - The discovery playbook

The JD says *"take the time to understand a customer's business challenges, ask thoughtful
questions."* This is the skill you are least practiced at, and it is the most learnable. Discovery
is a sequence, not a personality trait.

### The five questions that open any engagement

1. **"What decision are you trying to make that you can't make today?"**
   Forces a decision rather than a feature. If they cannot name one, the project has no owner.
2. **"Who's going to use this, and what do they do today instead?"**
   The current workaround tells you the real requirement. A spreadsheet somebody maintains by hand
   is a specification.
3. **"When this is working, what number moves?"**
   Directly serves the JD's "fully measurable outcomes." Also smokes out projects that exist for
   optics.
4. **"Where does this data live now, and who owns it?"**
   Ownership predicts your timeline more than volume does. The technical work is rarely the delay;
   getting access is.
5. **"What's already been tried here, and why didn't it stick?"**
   The single highest-value question. There is nearly always a predecessor project. Understanding
   why it failed - usually adoption or trust, not technology - saves you from repeating it.

### Follow-up moves

- **When a stakeholder gives a vague metric:** *"When you say 'better pricing,' what would you look
  at in three months to know it worked?"*
- **When two stakeholders disagree in the room:** do not resolve it live. *"I want to make sure I
  capture both of those - can I write up the two definitions and bring back what each one implies
  for the build?"* Then produce the artifact. That is exactly what your business-rule matrix was.
- **When someone asks for something technically unwise:** never lead with no. *"I can build that.
  Here's what it costs us later - and here's a version that gets you the same decision without
  that cost. Which do you want?"* Trade-offs, not refusals.
- **When you don't know:** *"I don't know - let me find out and come back today."* Then come back
  today. Consultants build trust on responsiveness far more than on omniscience.

### The thing nobody tells you about discovery

**Write down what you heard and send it back within 24 hours.** "Here's what I understood, here's
what I'm assuming, tell me where I'm wrong." Most engineers skip this. It catches misunderstandings
while they are cheap, it makes you look organized, and it creates a written record when scope
drifts later. You already have the instinct - the HLD and the business-rule matrix are the same
move.

---

## Part 4 - Live scenarios

Work these out loud before the interview. Expect at least one in a client-facing round.

---

### Scenario A - "The CEO wants AI"

> *A PE firm's portfolio company CEO says they want to "use AI" for sales. No further detail. You
> have a 45-minute call with him and his VP of Sales. Go.*

**How to work it:**

Do not propose anything in the first twenty minutes. Run discovery.

Ask what the sales team does manually today that they wish they did not. Ask what the VP looks at
every Monday morning. Ask what a rep does between getting a lead and first contact. Somewhere in
there is a real task - usually research, prioritization, note-taking, or proposal drafting.

Then narrow, out loud, so they see the reasoning:
> "From what I'm hearing, the highest-leverage thing isn't a model at all - it's that your reps
> spend an hour a day researching accounts before they call. That's a well-defined task with a
> reviewable output, which makes it a good first candidate. Something like forecast accuracy would
> be a much worse first project - the data isn't clean enough yet and you wouldn't be able to tell
> for two quarters whether it worked."

**What you are being scored on:** that you did not immediately architect something. That you scoped
to something measurable in weeks. That you declined a worse project *with a reason*.

**Then land the measurability point,** since the JD says it twice:
> "Before we build, I'd want to agree what we measure - time per account before and after, and
> whether the output is good enough that reps actually use it without rewriting it. Adoption is the
> real risk on this kind of tool, not accuracy."

---

### Scenario B - Post-acquisition consolidation (your home turf)

> *A portco just acquired a competitor. The PE firm wants combined revenue reporting in 60 days.
> Both companies have their own CRM and ERP.*

**How to work it:**

This is Story 1. Lead with the blocker, because naming it early is the credibility move:

> "The thing that will actually determine whether 60 days is realistic isn't the reporting layer -
> it's that the same customer exists in both systems with no shared key, and probably under
> different names, different legal entities, and different hierarchies. Until that's resolved, any
> combined revenue number double-counts or drops customers, and it'll be wrong in a way that's hard
> to see."

Then the plan, in the order you would actually do it:
1. Inventory both estates - what systems, who owns them, who can grant access. Access is the
   critical path, not engineering.
2. Agree the definitions *first*. Does "revenue" mean the same thing in both companies? Fiscal
   calendars, recognition timing, what counts as a customer. Two companies almost never match here,
   and this is where a 60-day project quietly becomes 120.
3. Resolve identity - customers first, then products, then people. Track the unmatched residue as a
   visible number from day one, and build the manual-override path immediately.
4. Land a combined read layer with the residue exposed rather than hidden.
5. Reporting last, and it is the easy part.

**The senior move - manage the deadline honestly:**
> "I'd want to be straightforward about the 60 days. I can almost certainly get you a combined
> revenue number that's directionally right and honest about its own gaps in that window. A number
> finance will close the books on is a different bar, and that usually depends on how fast we get
> agreement on definitions - which is their decision more than my throughput."

That is expectation management, which the JD explicitly names. **Do not accept an unrealistic
deadline silently in an interview scenario.** Being willing to say this is a large part of what
they are testing.

---

### Scenario C - The number is wrong

> *Two weeks after go-live, the CFO emails: "Your dashboard says $4.2M, our system says $3.9M.
> What's going on?" Copied to the PE operating partner.*

**How to work it:**

Sequence matters more than technique here.

1. **Acknowledge fast, before you know the answer.** Within the hour: "I see it, I'm on it, I'll
   have an explanation by end of day." Silence while you investigate is how trust dies.
2. **Assume you are wrong until proven otherwise.** Even if you suspect their system is the one at
   fault, never open with that.
3. **Reconcile stepwise, not all at once.** Same date range? Same entities included? Same inclusion
   rules - are house accounts, test records, and intercompany in or out? Same recognition timing?
   Same currency handling? The difference usually appears at one specific step, and finding *which*
   step is the whole job. You did exactly this at Innova: cross-system measure comparisons
   reconciling the same number computed from two systems, and a negative-margin defect traced from
   the report back through FP&A to the EBS source.
4. **Report the cause in their language.** "The $300K difference is 47 placements where the
   commission was manually adjusted in your system after the load. Our number reflects the system
   of record; yours reflects the adjustments. Both are 'right' - they answer different questions.
   Here's what I recommend we do about it."
5. **Then fix the class of problem, not the instance.** Governed manual overrides with a
   `manual_load_flag` propagated onto the fact - so an adjustment stays traceable in the report that
   shows it. That is a real thing you built.

**What is being scored:** speed of acknowledgment, refusal to blame, structured reconciliation, and
that you fixed the category rather than patching one number.

---

### Scenario D - The client's engineer is hostile

> *You're deploying into a portco. Their lead engineer clearly resents the consultants and says
> your architecture won't work in their environment.*

**How to work it:**

> "First - he's often right. He knows the environment and I don't, and 'it won't work here' usually
> encodes something real that nobody wrote down. So I'd want the specifics: what breaks, where has
> this pattern failed here before.
>
> Second, the underlying thing is usually not technical. Consultants arrive, build something, leave,
> and his team inherits it. That's a completely reasonable thing to be defensive about. So I'd want
> him shaping the design early rather than reviewing it late, and I'd want it to be obvious that
> he'll be able to run and change this after I'm gone - which means his conventions, his stack where
> possible, and documentation aimed at his team rather than at my deliverable list.
>
> If he's technically wrong about a specific point, I'd rather demonstrate it than argue it - build
> the small version, run it in their environment, look at it together."

**What is being scored:** you do not treat the client's engineer as an obstacle, and you understand
that handover is the actual product. The Reddit comment about FDEs being "rebranded support" who
play telephone with product engineers is the failure mode here - the answer is co-design, not
translation.

---

### Scenario E - Scope creep mid-engagement

> *Three weeks into a six-week build, the VP of Ops asks you to "also just add" inventory data.*

**How to work it:**

Never "no." Never silent "yes."

> "Let's look at what that means. Adding inventory means a new source system, its own identity
> problem against the customer and product dimensions, and probably its own definition argument
> about what counts as on-hand. That's not a small add - realistically it's two to three weeks on
> its own.
>
> So here are the options: we push the delivery date, we swap it for something currently in scope,
> or we finish what we agreed and take inventory as a phase two with its own timeline. My
> recommendation is the third - the pricing work we're doing is close to landing and it loses value
> if it slips. But it's your call and I'll write up whichever you pick so we've both got the same
> understanding."

**Three moves:** make the cost visible, present options rather than a verdict, recommend one, and
write it down. That is consulting.

---

## Part 5 - Whiteboard patterns

If asked to design something live, use this order every time. It reads as senior regardless of the
problem.

1. **Restate the problem and the decision it serves** - 30 seconds. "So the goal is that the sales
   VP can see accurate pipeline by rep across both companies. Let me make sure that's right before
   I draw anything."
2. **Ask about constraints before drawing** - data volume, refresh expectation, who maintains it,
   what already exists that you must live with. *"What's already in place that I should be building
   into rather than around?"*
3. **Draw the boxes: sources → landing → modeled → serving → consumers.** Simple, and it is what
   the work actually is.
4. **Name the hard part out loud and put your time there.** "The interesting problem here isn't the
   pipeline, it's that these two customer lists don't share a key." Identifying the hard part
   correctly is most of what is being scored.
5. **State one trade-off you are consciously accepting.** "I'm choosing daily batch over streaming
   because the decision it feeds is weekly. If that changes, this design changes."
6. **Say what you would measure to know it works.**

**If you get stuck:** say so and think out loud. *"I'm weighing two options here - let me talk
through both."* An FDE who visibly reasons under uncertainty is more valuable than one who bluffs,
because bluffing in front of a client is the thing that loses accounts.

---

## Part 6 - Fast recall table

| If they ask about... | Go to... |
|---|---|
| External clients / client-facing experience | **Story 8** - client-side finance and audit teams, Privia Health, HGC programs, the Volt company boundary |
| Hard technical problem | Entity resolution - Story 1 |
| Conflicting stakeholders | Revenue attribution - Story 2 |
| Ambiguity / shifting requirements | Six months of finance requirements - Story 3 |
| Post-merger / integration | Two-tenant takeover - Story 4, and the roll-up positioning |
| RAG / LLM work | FinSights - Story 5, plus the reranking regression |
| Agents / orchestration / MCP | LangGraph system - Story 6 |
| Trade-offs / judgment | Bullhorn display name vs EBS key - Story 7 |
| Failure / what went wrong | Reranking ablation, or the forced recursion failure test |
| What you'd do differently | Auto date/time left on - 115 generated date tables, 520 MB PBIX. Real, specific, and shows you audit your own work |
| Scale limitations | FinSights is single-service, no VPC/ALB/multi-tenant. Name it yourself |
| Teaching / enablement | 42-topic Power BI + 33-topic SQL curriculum; HLD v1.3; 35+ interviews conducted |

---

## Part 7 - The night before

- Re-read Part 2 of [[00 - FDE Core Positioning and Screen Prep]] - the roll-up positioning is the
  single thing to get right.
- Say Story 1 and Story 6 out loud, timed. Ninety seconds each, exec version.
- Have your four questions written on paper in front of you.
- Have your comp number decided and rehearsed so it comes out flat and unhesitating.
- Know which location this req is actually for.
- One line to keep in your head: **you are not underqualified, you are unproven on one axis, and
  the axis is the one this job exists to teach.**
