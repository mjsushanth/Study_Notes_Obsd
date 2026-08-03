
# S02f - Serving and Production: Processes, Health, Cost

Prerequisites: [[S02b - Networking - Sockets, DNS, Bridges, Discovery]] (the
four-tuple, listening sockets),
[[S02d - ECS Anatomy - Tasks, Services, Fargate]].

Answers points **10**, **11**, **15**, **17**, **18**, **19**.

Point **17** is the deepest question in your list, so it goes first and gets the most
room.

---

## 1) How a request finds a process (point 17)

Your question: *a backend contains multiple processes spawned - who gets which
process, who writes the code that links a process to a consumer, how is it mapped?*

**The answer that surprises everyone: nobody writes that code. The kernel does it,
and it is only about six system calls deep.** Let me build it from the bottom.

### The four calls

```c
fd = socket(AF_INET, SOCK_STREAM, 0);   // create an endpoint object
bind(fd, 0.0.0.0:8000);                 // claim an (address, port) in this netns
listen(fd, backlog);                    // become passive; create the accept queue
conn = accept(fd);                      // remove one completed connection
```

`listen()` is the interesting one. It turns the socket **passive** and creates a
queue. From that moment the *kernel* - not your program - completes incoming TCP
three-way handshakes and parks each finished connection in that queue. Your process
could be busy, asleep, or garbage-collecting; connections still land in the queue.

`backlog` is the queue's depth. Overflow it and the kernel drops or refuses new
connections - which is what "connection refused under load" usually is, rather than
anything your application chose.

`accept()` removes one connection from the queue and returns a **new** file
descriptor with its own full four-tuple. The listening socket keeps listening.

So a "server" is: one passive socket, plus a loop that pulls connections off a
kernel-managed queue.

### The pre-fork model, and who decides

Now the actual answer to "who gets which process."

```
   master process
     socket() -> bind(:8000) -> listen()          ONE listening fd
     |
     fork() fork() fork() fork()
     |        |        |        |
   worker1  worker2  worker3  worker4     each INHERITS the same fd
     |        |        |        |
   accept() accept() accept() accept()    all blocked on the same queue
```

`fork()` copies the file-descriptor table, so every worker holds a descriptor
pointing at the *same* kernel socket object. All of them call `accept()` on it and
block.

A connection arrives. The kernel completes the handshake, queues it, and **wakes
exactly one** blocked worker to receive it. Which one? The kernel's choice - in
practice roughly last-in-first-out on the wait queue, not a fair round-robin.

That is the entire mapping. There is:

- no registry of which worker handles what
- no dispatcher process
- no code you write to route a request to a worker
- no coordination between workers at all - they never talk to each other

**The kernel's accept queue is the load balancer.** This is the "software paradigm
scale" insight you were reaching for: distribution at the process level is a kernel
primitive, not an application concern. Load balancers (ALB, nginx) exist to
distribute across *machines*, because the kernel's trick only works within one
kernel.

Historical footnote worth knowing: waking *all* waiters on every connection was the
classic **thundering herd** problem. Modern Linux wakes one for `accept()`. The
newer `SO_REUSEPORT` option takes a different approach - each worker gets its own
listening socket on the same port, and the kernel hashes the connection's four-tuple
to pick a queue. That gives more even distribution, which matters at high connection
rates.

### The layer above: WSGI, ASGI, and what uvicorn is

Your application does not call `accept()`. Two layers sit between.

- **The application server** owns the socket and the worker processes, parses HTTP,
  and calls your code through a standard interface. `uvicorn`, `gunicorn`,
  `hypercorn`.
- **The interface standard** is that calling convention. **WSGI** is the old
  synchronous one (Flask, Django); **ASGI** is the async successor (FastAPI,
  Starlette), which additionally supports WebSockets and long-lived connections.

`uvicorn --workers 4` makes uvicorn itself the master: bind, listen, fork four
workers, each running an asyncio event loop.

### Three concurrency models, and the honest tradeoffs

| Model | Parallelism | Cost per unit | Fails when |
| :-- | :-- | :-- | :-- |
| **Processes** | true (separate GIL each) | ~50-100 MB RSS | memory-bound; caches duplicated N times |
| **Threads** | I/O only (GIL, per S01 §5) | ~8 MB stack | CPU-bound work serialises |
| **Async** | I/O only, one thread | ~KB per task | **any blocking call freezes everything** |

That last row is the trap in async servers. An `async def` handler that performs a
*blocking* call - a synchronous `boto3` request, `pl.read_parquet`, a `time.sleep` -
stops the entire event loop. Every other in-flight request on that worker stalls.
FastAPI mitigates this: a handler declared plain `def` (not `async def`) is run in a
thread pool, so blocking code is safe there. Declaring a handler `async def` and then
blocking inside it is the single most common FastAPI performance bug.

### What this means for FinSights, concretely

Your `backend.Dockerfile` runs:

```
uvicorn backend.api_service:app --host 0.0.0.0 --port 8000 --workers 1
```

**One worker.** So there is exactly one process, one event loop, and one accept
loop. Combined with a 30-50 second query built from three Bedrock variant calls,
four S3 Vectors queries, and a synthesis call, the practical consequences are:

1. **Concurrency is effectively one**, or a small number if the handler is a plain
   `def` running in the thread pool. Worth checking which it is - it determines
   whether a second visitor waits or is served.
2. **Scaling up is not free.** Each additional worker is a separate process with its
   own copy of the caches. If you cache the 344.5 MB table per process, four workers
   need roughly 1.4 GB just for that table. **Worker count and memory are coupled**,
   and this is exactly why the caching design in
   [[S02d - ECS Anatomy - Tasks, Services, Fargate]] and the task sizing interact.
3. **This is fine for your use case.** A demo serving one visitor at a time does not
   need more. But it is the reason Cloud Map's "independent scaling" was never
   reachable - there was nothing to distribute to.

---

## 2) Streamlit: the session-affine server (point 10)

[[S02a - Foundations - Processes, Namespaces, Containers]] §6 introduced a third
grade of statefulness that nobody teaches. Streamlit is its textbook case.

Streamlit's model: the browser loads a page which opens a **WebSocket** back to the
server. A WebSocket starts as an HTTP request carrying `Upgrade: websocket`; after
the handshake the connection stops being request/response and becomes a persistent
bidirectional byte stream. It stays open for the life of the tab.

The server keeps per-session state in memory (your `st.session_state`, widget
values, chat history) keyed to that connection, and **re-runs your entire script
top-to-bottom on every interaction**, diffing the resulting widget tree and pushing
updates down the socket.

So a Streamlit session is:

- **long-lived** - minutes to hours, not milliseconds
- **stateful in process memory** - session state is not in a database
- **pinned** - that browser tab must keep talking to *that* process. Route the next
  message elsewhere and the state is not there.

This is why Streamlit behind a load balancer requires **sticky sessions**, and why
each session consumes real memory for as long as the tab is open.

And it is the complete, unarguable reason your application could never have run on
Lambda.

---

## 3) Lambda's execution model, and why RAG fits it badly (points 10, 11)

Lambda's contract: you give AWS a function. It runs the function in response to an
event and gives you no control over the process lifecycle.

```
  event -> [ is a warm execution environment available? ]
             |                              |
            no                             yes
             |                              |
        COLD START                      WARM START
        - provision a micro-VM          - reuse the process
        - load your code                - your module-level state
        - run module-level init            is still there
        - run the handler               - run the handler
             |                              |
             +---> respond ---> environment kept alive ~5-15 min ---> frozen/destroyed
```

Two properties dominate everything:

- **You do not own the process.** It can be destroyed between any two invocations.
  Anything you cached in module scope may or may not still exist. You may not assume.
- **There is no listening socket.** Lambda has no `accept()` loop you control, no
  persistent connection, no WebSocket. The runtime hands you one event and takes the
  response back.

That second point is fatal for Streamlit, independently of everything else.
**A long-lived WebSocket server cannot exist inside a request/response function.**
Not with a bigger package, not with a container image, not with more memory. Your
instinct in point **10** - "Streamlit is a long-living server, Lambda is
sleep/invoke/wake/work/sleep" - is exactly right, and it is a structural
impossibility rather than a tuning problem.

### The packaging limits, and the one you were fighting

| Packaging | Unzipped size limit |
| :-- | :-- |
| Zip (function + layers combined) | **250 MB** |
| **Container image** | **10 GB** |

Your `.samignore` - 4,855 bytes of surgical exclusions - was a fight against the
250 MB limit. My estimate of your dependency set (pyarrow ~120 MB, botocore ~90 MB,
polars, numpy, nltk) puts you at roughly 280-380 MB, so you would have lost that
fight.

But you had already sketched the container-image route in your own notes, and you
already had Docker and ECR working. **The wall you spent the most effort on was one
packaging-mode switch away from irrelevant.** The genuinely interesting detail: the
250 MB limit is never mentioned once in any of your artifacts. You were fighting it
by instinct without naming it - which is precisely why naming constraints explicitly
is worth the discipline.

### Point 11, which is the sharper objection

Your point 11 is the better argument, and it is the one that would have killed
Lambda even with container images:

> a Lambda that rebuilds a requirement to gather retrieval-based embedding ID,
> sentence ID, and needs to talk to a data table or parquet, would restart that on
> every cold start.

Correct, and here are the numbers. Your `SentenceExpander.__init__` loads the Stage 2
meta table - **64.8 MB compressed in S3, 344.5 MB resident** (your own measurement) -
and then does a `sentence_pos` extraction pass over all 469,252 rows. Plus four more
index structures.

In a long-running container that is a **one-time** cost amortised over every request
for the container's whole life. In Lambda it is a **per-cold-start** cost, and you
control neither how often cold starts happen nor when. Lambda's `/tmp` (512 MB by
default) is the standard workaround, and notice what it is: **local disk caching to
compensate for not being allowed to keep memory.** That is fighting the platform.

Two more concrete misfits from your own artifacts:

- `template.yml` sets `Timeout: 30`. Your real path is three Bedrock variant calls
  plus four S3 Vectors queries plus synthesis, serially - well past 30 s.
- API Gateway's default integration timeout is ~29 s. So even raising the Lambda
  timeout to 900 s would not give you a synchronous HTTP query endpoint; you would
  have to go asynchronous, which destroys the request/response UX the design assumed.

**Verdict:** Lambda is excellent for short, stateless, event-driven, bursty work.
Your workload is long-running, heavy-init, and served over persistent connections.
The pivot to ECS was correct - and worth noting, *you reasoned your way to the right
answer before testing it*, which is why there are no error logs in `lambda_assets/`.
There was never a deployment to fail. That is a good outcome, not a gap.

---

## 4) Health checks: why `/health` is a liar (point 15)

### The three kinds, which is the whole lesson

Production systems need **three different** questions answered, and collapsing them
into one endpoint is the classic mistake:

| Probe | Question | Action on failure | Must it check dependencies? |
| :-- | :-- | :-- | :-- |
| **Liveness** | Is the process wedged? | **restart it** | **NO - never** |
| **Readiness** | Can it serve traffic *now*? | **remove from the pool** | **YES** |
| **Startup** | Is it still initialising? | wait; suppress liveness | no |

The rule in that last column is the one that causes real outages when violated:

> **A liveness probe must never check its dependencies.**

Why: suppose liveness checks S3. S3 has a brief regional blip. Every task fails
liveness simultaneously. The orchestrator restarts all of them. They come back cold,
all hammer S3 at once, fail again, and restart again. **You have converted a
30-second dependency blip into a self-inflicted total outage with a restart storm.**

Readiness is the probe that *should* check dependencies, because its remedy is
"stop sending traffic," which is safe and self-healing. When S3 recovers, readiness
passes and traffic resumes with no restarts.

Startup probes exist because a slow-booting process would otherwise fail liveness
during normal initialisation and be killed forever. ECS's equivalent is
`healthCheckGracePeriodSeconds`; Docker's is `--start-period`.

### Now, your endpoint

`api_service.py:144-157`:

```python
return HealthResponse(status="healthy",
    model_root_exists=config.model_pipeline_root.exists(),
    aws_configured=None,                 # hardcoded
    timestamp=datetime.utcnow().isoformat() + "Z")
```

It returns `"healthy"` **unconditionally** - there is no code path that returns
anything else - and `aws_configured` is a hardcoded `None`. The docstring is honest
about it: *"AWS credentials are validated by orchestrator on first query."*

So it is a **liveness** probe wearing a **readiness** probe's name. As liveness it is
actually fine and correctly designed: cheap, local, no dependencies. The problem is
that it is *used* as readiness - by both Dockerfiles' `HEALTHCHECK`, and by anything
that would put this behind a load balancer.

The practical failure mode: a task with a completely broken IAM task role - no
Bedrock access, no S3 - passes health checks forever, reports `Up (healthy)`,
receives traffic, and fails every single user query. **This is precisely how the
December `s3vectors:QueryVectors` denial happened in production rather than at
deploy time.** The task was, by its own report, perfectly healthy.

### The proper production workflow

```
  GET /health/live    -> process responds. no I/O, no dependencies. cheap.
                         orchestrator restarts on failure.

  GET /health/ready   -> can I actually serve?
                         - config loaded, components constructed
                         - a CHEAP dependency probe: STS get-caller-identity,
                           or s3:HeadObject on one known key
                         - result CACHED ~10-30s so probes don't hammer AWS
                         load balancer removes from pool on failure.

  deploy smoke test   -> POST /query with a known question, assert a real answer.
                         this is the ONLY thing that proves the task role works.
```

That last line is the operational conclusion, and it is what your deploy script must
do. `ecs wait services-stable` proves only that ECS reached steady state. It says
nothing about whether the application can answer a question. **The old pipeline had
no smoke test of any kind** - which is why it could report "DEPLOYMENT SUCCESSFUL"
(an unconditional `echo`) over a backend that could not retrieve.

The general principle, worth carrying beyond this project:

> A health check is a **promise about behaviour**. If it does not exercise the
> dependency, it does not make a promise about it. And a green check that promises
> nothing is worse than no check, because it manufactures false confidence.

---

## 5) Public product versus public API (point 18)

You are right and I was imprecise, so let me restate it properly.

**The RAG was meant to be public. That was the point, you shipped it, and you
presented it.** A public frontend on 8501 is not a security finding - it is the
product working as designed. I framed it as a hole and that was wrong.

The distinction I should have drawn is between two separate doors:

| Door | Status | Assessment |
| :-- | :-- | :-- |
| Frontend, 8501 | public | **the product.** Correct by design |
| Backend API, 8000 | public | **incidental.** Nobody chose this |

Why the second still matters, in one concrete sentence: anyone who found the task's
IP could `POST /query` directly, bypassing your UI, and **each call spends real
Bedrock money at roughly $0.02-0.03**. There is no authentication, no rate limit, and
no quota on that endpoint.

So the risk was never disclosure. It was **an unauthenticated, unmetered, paid
endpoint on the open internet.** For a demo that ran briefly and was then torn down,
the practical exposure was near zero. As a standing architecture it is the kind of
thing that produces a surprising bill.

And the pleasing part: the one-task design closes that door **for free**, without
compromising the public product at all. Port 8000 is never opened, because
`localhost` needs no exposure. The frontend stays exactly as public as you want it.
Same product, one fewer unmetered door - which is a better outcome than either of us
started from.

The general lesson for public-facing ML: **the expensive endpoint needs a different
security posture from the page in front of it.** Real systems put the paid call
behind auth, a rate limit, and a spend cap, and expose only the UI. That is a
genuinely useful thing to be able to say about your own architecture in an
interview.

---

## 6) Scale to zero, destroy, and reproducibility as a property (point 19)

### What "down" actually costs

Fargate bills **only running tasks**. So a service at `desired-count 0` costs nothing
in compute, and the cluster, task definitions, security groups, and IAM roles are all
free records, permanently.

| State | Monthly cost | Time to serving |
| :-- | :-- | :-- |
| running, 1 vCPU / 4 GB | ~$42.53 | - |
| running on Fargate Spot | ~$12.80 | - |
| `desired-count 0` | **~$0.06** (ECR only) | ~2 min |
| fully destroyed | **$0.00** | ~10 min (rebuild + push) |

Mapping it to what you already know from Docker:

```
  up      ==  docker compose up -d          desired-count -> 1
  down    ==  docker compose stop           desired-count -> 0
  destroy ==  docker compose down --rmi all delete services, cluster, ECR images,
                                            log group, SG, IAM roles
```

So `destroy` buys you about six cents a month over `down`. On pure economics it is
not worth the eight extra minutes.

### But your instinct is right for a completely different reason

You said you do not mind tearing down and waiting, *as proof of a reliable
zero-to-reproduction build*. That is the correct reason to want it, and it is worth
stating as a principle:

> **The only way to know your infrastructure code is complete is to destroy
> everything and rebuild from nothing.**

Anything you forgot to declare - a hand-edited security group rule, a task
definition that only exists in the console, an IAM policy someone added during an
incident - survives every incremental deployment and is **invisible** until the
rebuild exposes it.

This is not hypothetical for you. It is exactly what happened to FinSights:

- the task definitions existed only in AWS, so they died with the account
- port 8501 must have been opened by hand in the console, since no code opens it
- the `s3vectors` policy fix was applied live, then back-ported afterward

Each of those was invisible while the deployment kept running, and each was fatal the
moment you needed to rebuild. **A destroy-and-rebuild cycle is the test that catches
all three.** Six cents a month is the wrong frame; the right frame is that
`destroy` + `up` is your integration test for the deployment itself, and the only
honest one.

The industry phrasing is **"pets versus cattle."** A pet is a machine you nurse,
patch by hand, and cannot replace - and every hand-fix makes it more irreplaceable.
Cattle are identical, disposable, and rebuilt from a declaration. The FinSights ECS
deployment was a pet, and it died with its account. Making it cattle means the
declaration in the repository is *sufficient*, and the only proof of sufficiency is
that you regularly throw it away.

Which reframes the earlier lesson from
[[S02d - ECS Anatomy - Tasks, Services, Fargate]]. Declaring the task definition in
the repo is necessary. **Destroying and rebuilding is how you verify the declaration
is complete.** One is the claim; the other is the test.

---

## 7) Carry-forward

1. `socket -> bind -> listen -> accept`. `listen()` hands connection queuing to the
   kernel; **the kernel's accept queue is the load balancer within one machine.**
2. Pre-fork workers all inherit one listening fd and all `accept()` on it. Nobody
   writes routing code. Load balancers exist to cross *machine* boundaries.
3. Processes give true parallelism at ~50-100 MB each; async gives cheap I/O
   concurrency but **any blocking call freezes the loop.** Worker count and cache
   memory are coupled.
4. Streamlit is session-affine: long-lived WebSocket, state in process memory,
   pinned to one process. Hence sticky sessions, and hence never Lambda.
5. Lambda gives you no listening socket and no ownership of the process. Zip is
   250 MB, container images 10 GB - but the real misfit was per-cold-start re-init of
   a 344.5 MB table.
6. **Liveness must never check dependencies; readiness must.** Your `/health` is a
   correctly-designed liveness probe being misused as readiness. Only a real query
   proves the task role works.
7. The public frontend was the product; the separately public paid API was the
   accident, and co-location closes it for free.
8. Down is ~six cents. **Destroy-and-rebuild is not a cost optimisation - it is the
   integration test for your infrastructure code.** Pets die with their accounts.
