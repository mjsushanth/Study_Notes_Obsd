
# S02 INDEX - Curriculum and Answer Map

Companion to [[S02 - Design Learn from FinSights RAG]], which holds your raw
reflection: 20 points, ~50 embedded questions.

Your 20 points are not 20 topics. They are about **eight fundamentals wearing 20
costumes**. The six teaching files below are ordered so each one only needs what came
before it. Read them in order; skipping ahead will feel harder than it is.

---

## Study order

| # | File | Covers | Read time |
| :-- | :-- | :-- | :-- |
| 1 | [[S02a - Foundations - Processes, Namespaces, Containers]] | processes, namespaces, cgroups, images/layers, statelessness, volumes | 25 min |
| 2 | [[S02b - Networking - Sockets, DNS, Bridges, Discovery]] | four-tuple, `localhost`, DNS, bridges/veth, embedded DNS, discovery, load balancers | 30 min |
| 3 | [[S02c - AWS Substrate - EC2, VPC, Subnets, NAT, ECR]] | EC2, VPC, subnets, route tables, NAT, security groups, ECR, CloudWatch, Cloud Map, idempotency | 30 min |
| 4 | [[S02d - ECS Anatomy - Tasks, Services, Fargate]] | cluster/task-def/task/service, Borg history, billing floor, ordering, Fargate, the 3-way decision | 30 min |
| 5 | [[S02e - Credentials and Config - Injection, IMDS, Loaders]] | the 3 injection channels, env vars, 12-factor, link-local credentials, your loader, detection patterns | 30 min |
| 6 | [[S02f - Serving and Production - Processes, Health, Cost]] | socket->worker mapping, concurrency, Streamlit, Lambda, health checks, public API, teardown | 35 min |

About three hours if you read properly. The three highest-value sections, if you are
short on time: **S02b §7** (why DNS cannot load-balance), **S02f §1** (how a request
finds a process), **S02f §4** (liveness vs readiness).

### Then read the walkthrough

| 7 | [[S03 - Systems Walkthrough - Deploying a RAG Service to AWS]] | the same material applied to a real deployment, with four diagrams and every number measured | 50 min |

### Then the three deeper notes, in this order

| 8 | [[S02g - Concurrency and Shared State - Threadpools, Thread Safety, the Audit]] | why a single-worker FastAPI process still has real concurrency; boto3 client vs resource vs Session thread safety; the AST statelessness audit and its blind spots | 30 min |
| 9 | [[S02h - Measurement as a Design Practice]] | six measurement methods with the numbers they produced, what each tool structurally cannot see, and three times my own measurement was the broken thing | 35 min |
| 10 | [[S02i - Higher-Level Design Principles from a Real Deployment]] | nineteen principles, each grounded where it earned its keep: unrepresentable states, convergence, noise floors, provenance labelling, failure asymmetry | 35 min |

These three are the abstract layer. S02g is the most concrete (it ends with a verified answer
for this codebase and a reusable script). S02h is the one to reread before any future
performance work. S02i is the one to skim when starting a *different* project.

| 11 | [[S02j - Empirical Methods and Findings Catalogue]] | every measurement and study this project has run, with exact numbers, `path:line` citations, and provenance labels on each figure | reference |

S02j is a **reference document, not a read-through**. It is the catalogue: the embedding
transport/determinism study, the token-accounting and cost reconciliations, the full reranking arc
(five linked studies), the memory-explosion investigation, and every measurement from the
deployment session. It also carries two things worth knowing before trusting any number in this
repo: a **consolidated contradiction table** (Part 6) and an explicit list of **what has never
been measured** (Part 7).

Canonical copy lives in the repo next to the scripts it documents:
`ModelPipeline/finrag_ml_tg1/investigation_analysis/EMPIRICAL_METHODS_AND_FINDINGS.md`.

S02a-f teach the mechanisms one at a time. **S03 is what happened when they were all used
at once**, on 2026-07-31: FinSights deployed to ECS Fargate, verified with a real query,
destroyed, and rebuilt from the repository alone. It is organised around the five questions
every deployment must answer — packaging, placement, wiring, identity, lifecycle — and its
second half looks at the deployment *tool* as an exercise in object design (dependency
injection, cached-client lifetime, subprocess-as-method, idempotency as convergence).

Read S03 after S02b and S02d at minimum. It re-states the load-bearing ideas, so it works
standalone, but it lands much harder if the mechanisms are already familiar.

---

## Two corrections to the record, before you study

**1. Sevalla and Lambda were never deployed. ECS was the real deployment, and it was
public.** I had earlier read your repo as "ECS attempt abandoned in favour of
Sevalla," because `.claude/CLAUDE.md` says *"Deployed on Sevalla."* That line is
wrong. You wrote Sevalla code and docs, never got it working, and dropped it; Lambda
never got past scaffolding. **ECS Fargate is the only thing that ever served real
traffic.** The notes are written on that basis. Worth fixing that line in
`CLAUDE.md`, since it will mislead anyone who reads the repo.

**2. Your point 18 corrected me, and I was wrong.** I called the public backend a
security hole without distinguishing it from the public *product*. The frontend being
public was the whole point and you shipped it. What remained worth flagging was
narrower: the paid `/query` endpoint was *separately* reachable, unauthenticated and
unmetered. Full treatment in **S02f §5** and **S02c §5**.

Your files are also restored: `backend.Dockerfile`, `frontend.Dockerfile`, and
`docker-compose.yml` are back in `finrag_docker_loc_tg1_aws/`, byte-identical to
`origin/main`, uncommitted.

---

## Answer map: your 20 points

| Your point | Short answer | Depth |
| :-- | :-- | :-- |
| **Header:** ECS, ECR, EC2, load balancers | orchestrator / image registry / rented VMs / reverse proxy | S02c §1,6; S02b §8; S02d §1 |
| **Header:** VPCs, runtimes, CloudWatch log groups, Cloud Map namespaces | rented network namespace / the layer that executes your code / log hierarchy with retention / managed DNS discovery = a Route 53 private zone | S02c §2,7,8 |
| **Header:** namespace, logical grouping, tasks, services, service discovery | namespace = *enforced* perception boundary; "logical grouping" = accounting label with **no** enforcement; task = co-scheduled container bundle; service = keeps N tasks alive; discovery = registration + resolution + health | S02a §2; S02d §1; S02b §6 |
| **1** embedded DNS, network bridges | bridge = software L2 switch; veth = virtual cable into the container's netns; Docker runs a resolver at **`127.0.0.11`** which makes compose service names resolvable. ECS has no equivalent - which is why `http://backend:8000` stops working | S02b §4,5 |
| **2** localhost vs Cloud Map vs ALB | three answers to one question, at three price points: shared netns (free) / DNS discovery (~$0.50) / reverse proxy (~$16.43). **Decision: option A.** DNS discovery solves churn but not distribution, so B's only benefit needs C | S02b §7; S02d §8 |
| **3** injecting keys into a "docker machine" | false premise - there is no machine and nothing is injected into a running container. Three channels, all decided at or before `exec`: build time, start time, run time | S02e §1-3 |
| **4** short-lived link-local credentials - how? | **your code does not do the looking; boto3 does.** Its credential chain includes an HTTP GET to `169.254.170.2`, an unroutable link-local address answered by the local ECS agent, returning credentials that expire in hours | S02e §6 |
| **5** does MLConfig fall back? | **Yes - and ECS needs zero code changes.** `ml_config_loader.py:56-61` returns `IAM_ROLE` before touching `.aws_secrets/`. One sharp edge: with no marker env var it raises rather than deferring to boto3's own chain | S02e §7 |
| **6** cross-service ordering; containers vs tasks | ECS *reconciles continuously*, so a one-time ordering promise would be a lie; components must tolerate absent dependencies and retry. **Within** one task, `dependsOn: HEALTHY` works - so option A preserves your `service_healthy` semantics | S02d §4,5 |
| **7** stateless, volumes, bind mounts | stateless = "destroying this instance loses nothing that mattered." Bind mount = host path grafted into the mount namespace (dev tool). Volume = engine-managed. **Fargate supports neither** - and you need neither | S02a §6,7 |
| **8** Fargate | the container programming model with **Firecracker micro-VM** isolation. Explains every quirk: slow starts, no volumes, no privileged mode, and **Docker `HEALTHCHECK` ignored** | S02d §6 |
| **9** VPCs, guard-and-check, NAT | public vs private is a property of the **route table**, not the subnet. Guard only the *non-idempotent* operation - your old workflow guarded role creation *and* policy attachment together, so a partial failure never self-healed | S02c §3,4,9 |
| **10** Streamlit vs Lambda; 250 MB vs 10 GB | Streamlit is **session-affine**: long-lived WebSocket, state in process memory, pinned to one process. Lambda gives no listening socket, so this is structurally impossible. Zip 250 MB / image 10 GB - and you fought the 250 MB wall without ever naming it | S02a §6; S02f §2,3 |
| **11** Lambda re-init per cold start | your sharper objection, and the one that kills Lambda even with container images. **64.8 MB compressed -> 344.5 MB resident**, plus a pass over 469,252 rows, re-paid per cold start. Lambda's `/tmp` trick is disk caching to compensate for not owning memory | S02f §3 |
| **12** environment-recognition loaders | genuinely a production pattern, with three failure modes yours shows mildly: detection by proxy, one variable with two meanings (`MODEL_PIPELINE_ROOT` also flips S3 streaming), and raising instead of degrading. Better: **explicit declaration over detection** | S02e §8 |
| **13** restore the deleted files | **done.** Recovered from the git object store, byte-identical to `origin/main`, `HEAD` untouched, left uncommitted | above |
| **14** no NAT gateway | correct call. IGW is **free**; NAT is ~$32.85/mo *per AZ* plus $0.045/GB. The general lesson: in small AWS systems the **fixed-price managed components dominate the bill** | S02c §4 |
| **15** `/health` is a liar | three probes, not one. **Liveness must never check dependencies** (or a blip becomes a restart storm); readiness must. Yours is a well-built liveness probe misused as readiness. Only a real query proves the task role | S02f §4 |
| **16** why the 0.25 vCPU floor; why containers live in tasks | Borg *alloc* -> k8s *pod* -> ECS *task*, all because production processes come in co-scheduled bundles (sidecars). The task is the **billing unit**, with a floor because each is a micro-VM. So task *count* drives cost | S02d §2,3 |
| **17** multiple processes - who gets which? | **nobody writes that code; the kernel does.** Workers inherit one listening fd, all `accept()` on it, the kernel wakes one. The accept queue *is* the load balancer inside one machine | S02f §1 |
| **18** the RAG was meant to be public | agreed, and my earlier framing was wrong. Public frontend = the product. The narrower issue was an unauthenticated, unmetered **paid** endpoint - which co-location closes for free | S02f §5 |
| **19** scale to zero; teardown as proof | down ~6 cents/mo, destroy $0. But the real reason is yours: **destroy-and-rebuild is the only test that proves your infrastructure declaration is complete.** Pets vs cattle - and FinSights was a pet that died with its account | S02f §6 |

---

## The eight fundamentals underneath

If you remember nothing else:

1. **A process's view of the world is a property of the process.** The kernel decides
   it. Namespaces enforce perception; cgroups enforce consumption.
2. **`localhost` means "this network namespace,"** not "this machine." Everything
   about co-located containers follows.
3. **DNS solves churn, not distribution.** Load balancing requires terminating
   connections.
4. **Public vs private is a route table property.** IGW free, NAT ~$33/mo.
5. **The task is the billing unit**, and it exists because production processes come
   in co-scheduled bundles.
6. **The SDK owns the credential chain**, so "use an IAM role" is a deployment change
   rather than a code change.
7. **The kernel's accept queue is the load balancer** inside one machine.
8. **Infrastructure that lives only in a console is not yours** - and destroy-and-rebuild
   is the only honest proof that your declaration is complete.

---

## The architecture, old and new

```
  WHAT SHIPPED IN DECEMBER (account 729472661729, now dead)
  ---------------------------------------------------------
   browser --> task public IP :8501  [frontend task]  0.25 vCPU / 0.5 GB
                                          |
                                     Cloud Map DNS
                                  backend.finsights.local:8000
                                          |
                                     [backend task]   512 CPU / 1024 MB  <-- would OOM
                                          |
                             Bedrock + S3 + S3 Vectors
                                          
   - 2 tasks (2 billing floors) + private hosted zone
   - port 8000 open to 0.0.0.0/0
   - task definitions existed ONLY in the console  <-- died with the account
   - no log group, no retention, no smoke test
   - static IAM user keys in GitHub secrets
   - ~$52/month


  WHAT WE BUILD NEXT (account 908877262866)
  -----------------------------------------
   browser --> task public IP :8501
                    |
   +----------------|--------------------------------------------+
   |  ONE Fargate task, 1 vCPU / 4 GB, one ENI, one port space   |
   |                                                             |
   |   [frontend container]  ---> http://localhost:8000          |
   |          8501 exposed          (shared network namespace,    |
   |                                 no DNS, no discovery)       |
   |   [backend container]                                       |
   |          8000 NEVER exposed                                 |
   |          dependsOn: backend HEALTHY                         |
   +-------------------------|-----------------------------------+
                             | IAM task role via 169.254.170.2
                             v
                  Bedrock (claude-haiku-4-5, cohere.embed-v4)
                  S3        (sentence-data-ingestion-mjs)
                  S3Vectors (finrag-embeddings-s3vectors)

   - 1 task, 1 billing floor
   - no static credentials anywhere
   - task definition declared in the repo
   - log group with 14-day retention
   - deploy ends in a real /query smoke test
   - up / down / destroy
   - ~$42.50/month running, ~$0.06 down, $0.00 destroyed
```

---

## Reference numbers (verified 2026-07-30)

| Thing | Value |
| :-- | :-- |
| Account / region | `908877262866` / `us-east-1` |
| S3 data bucket | `sentence-data-ingestion-mjs` |
| S3 Vectors bucket / index | `finrag-embeddings-s3vectors` / `finrag-sentence-fact-embed-1024d` |
| Stage 2 meta parquet | **64,781,290 bytes** compressed; **344.5 MB** resident |
| Stage 2 rows / vectors | 469,252 rows / 203,076 vectors |
| Fargate vCPU-hour | $0.04048 |
| Fargate GB-hour | $0.004445 |
| Hours per month | 730 |
| ALB | $0.0225/hr = ~$16.43/mo |
| NAT gateway | ~$0.045/hr = ~$32.85/mo + $0.045/GB |
| Internet gateway | $0.00 |
| ECR storage | $0.10/GB-month |
| Route 53 private hosted zone | ~$0.50/mo |
| Fargate ephemeral storage | 20 GB default, to 200 GB |
| Lambda zip / image limit | 250 MB / 10 GB |

Two things deliberately marked **UNVERIFIED**, because I did not test them:

1. **Polars S3 reads under a bare IAM task role.** `get_storage_options()` hands
   Rust's `object_store` only a region and lets it resolve credentials itself. It
   should support the ECS endpoint; I have not proven it. This would fail on the
   *first query*, not at startup, and it is the single most likely cutover breakage.
2. **Backend cold-start time.** Estimated 3-8 s to first `200 /health` from the
   import chain. Your Dockerfiles assume 40 s (`--start-period`), which is a
   comfortable envelope either way.

---

## What comes next in the project

Not yet built, in order:

1. **A construction profile** - time and memory for each component in the
   per-request chain, so the task's memory figure is measured rather than my estimate,
   and the caching order is evidence-based. Ties to your chosen option 2 on caching.
2. **The caching fix** in `orchestrator.py`, per the "expensive AND immutable" rule
   (the table you screenshotted into your reflection).
3. **The task definition JSON** in the repo, with the `healthCheck` restated (since
   Fargate ignores the Dockerfile's) and `AWS_EXECUTION_ENV` set explicitly.
4. **The least-privilege task role** - note the `us.` model prefix is a cross-region
   inference profile, so the policy needs the inference-profile ARN *plus* the
   underlying foundation-model ARNs in us-east-1, us-east-2, and us-west-2.
5. **`up` / `down` / `destroy`**, ending in a real `/query` smoke test.
