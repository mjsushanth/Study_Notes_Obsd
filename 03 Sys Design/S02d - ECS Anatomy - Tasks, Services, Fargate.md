
# S02d - ECS Anatomy: Tasks, Services, Fargate

Prerequisites: all of [[S02a - Foundations - Processes, Namespaces, Containers]],
[[S02b - Networking - Sockets, DNS, Bridges, Discovery]],
[[S02c - AWS Substrate - EC2, VPC, Subnets, NAT, ECR]].

Answers points **2 (the decision)**, **6**, **8**, **16**, and the header questions
*ECS*, *tasks and services*, *runtimes*.

---

## 1) The four objects, and the one that matters most

ECS (Elastic Container Service) is an **orchestrator**: it decides where containers
run, starts them, watches them, and replaces them when they die.

| Object | What it is | Nearest thing you already know |
| :-- | :-- | :-- |
| **Cluster** | a logical grouping of capacity. Free on Fargate. | a label; see "logical grouping" in S02a |
| **Task definition** | an immutable, versioned blueprint | your `docker-compose.yml` |
| **Task** | one running instance of a definition | a running container group |
| **Service** | a controller that keeps N tasks alive | `restart: unless-stopped` |

The **task definition** is the object to understand deeply, because it is the only
one that is a real artifact you author. It specifies:

- the container list, each with image, port mappings, environment variables,
  log configuration, and health check
- **task-level** CPU and memory (this is what you are billed for)
- the **task role** and the **execution role** - two different things, see §7
- network mode, ephemeral storage, container dependency ordering

Registering a task definition creates an immutable **revision**:
`finsights-task:1`, `:2`, `:3`. You never edit one. You register a new revision and
tell the service to use it. Rolling back is pointing the service at an older
revision - which is instant, and free, *provided the revision exists*.

### The single most important lesson from your December deployment

The old pipeline never had a task definition in the repository. It read the live one
out of AWS, patched two fields with `jq`, and re-registered it:

```bash
aws ecs describe-task-definition --task-definition finsights-backend-task \
  --query taskDefinition > task-def.json
jq '.containerDefinitions[0].image = $IMAGE' ... > new-task-def.json
aws ecs register-task-definition --cli-input-json file://new-task-def.json
```

Read that carefully. **AWS was the source of truth, and the repository was a
mutation script.** Every real decision - how much memory, which log group, which
IAM role, which environment variables - existed only as console state in an account
that no longer exists. When the account went away, the design went with it.

The principle, stated generally:

> **Infrastructure that exists only in a cloud console is not infrastructure you
> own. It is infrastructure you are renting the memory of.**

This is why the phrase "infrastructure as code" matters, and it is not about
Terraform specifically. It is about the task definition being a file you can read,
diff, review, and re-apply into an empty account.

---

## 2) Why containers live inside tasks: the history (point 16)

Your question is exactly the right one to ask: *why is there a wrapper around the
container at all?* Why not just "run this container"?

The answer is genuinely historical, and it comes from Google.

**Borg (Google, ~2003).** Google's internal cluster manager. Its schedulable unit
was not a single process but an **"alloc"** - an *allocation* of resources on one
machine, into which one or more related processes could be placed. The insight from
running that system for a decade was that **certain groups of processes must be
co-scheduled**: they must land on the same machine, share a network identity and
some filesystem, start and stop together, and be accounted for as one unit.

Why? Because of the **sidecar** pattern. Real production processes are almost never
alone. They come with:

- a log shipper reading the app's output and forwarding it
- a metrics exporter
- a proxy handling TLS, retries, or service-mesh routing
- a credential-refresher rotating short-lived secrets
- a config-reloader watching for changes

None of these make sense on a *different* machine from the app. A log shipper that
cannot read the app's stdout is useless. So the scheduler needs an atomic unit
**larger than one container**.

**Kubernetes (2014)** took this directly from Borg and called the unit a **pod**.
**ECS (2014-15)** made the same choice and called it a **task**. They are the same
idea with different names:

```
   Borg alloc   ==   Kubernetes pod   ==   ECS task
   
   a co-scheduled bundle of containers that:
     - land on the same host together
     - share a network namespace  (one IP, one port space)
     - can share volumes
     - are started, stopped, and replaced as a unit
     - are billed / accounted as a unit
```

So the container is the *packaging* unit and the task is the *scheduling* unit.
Conflating them is the most common beginner error in this space, and separating
them is the whole answer to point **6**.

### What sharing a network namespace buys - and this is the payoff

Because containers in one task share a network namespace (see S02a §2, S02b §2),
they share **one loopback interface and one port space**. Therefore:

- they reach each other at **`localhost`**
- they **cannot both bind the same port** - two containers on 8000 in one task is a
  startup failure, not an isolation success
- they present **one IP address** to the outside world

That first bullet is the entire mechanical justification for the recommended
FinSights design. `BACKEND_URL=http://localhost:8000` works not by configuration
but by construction.

---

## 3) The task is the billing unit - hence the 0.25 vCPU floor

Second half of point **16**. Fargate bills **per task**, on the task's declared CPU
and memory, per second, for as long as the task runs. Containers inside a task do
not have their own bill; they share the task's allocation.

And you cannot buy an arbitrarily small task. The valid Fargate combinations:

| Task CPU | Allowed task memory |
| :-- | :-- |
| 0.25 vCPU (256) | 0.5, 1, 2 GB |
| 0.5 vCPU (512) | 1-4 GB, in 1 GB steps |
| 1 vCPU (1024) | 2-8 GB |
| 2 vCPU (2048) | 4-16 GB |
| 4 vCPU (4096) | 8-30 GB |
| 8 vCPU (8192) | 16-60 GB, in 4 GB steps |

Why a floor exists: Fargate runs each task in its own micro-VM (§6). A VM has
irreducible overhead - a kernel, a guest agent, memory for page tables. AWS cannot
profitably sell you a 32 MB VM, and it would not work anyway.

**The consequence you must internalise:** *the number of tasks, not the number of
containers, drives your bill.* Two tasks pay two floors. One task with two
containers pays one.

Working the arithmetic at `us-east-1` on-demand rates - $0.04048 per vCPU-hour,
$0.004445 per GB-hour, 730 hours/month:

```
one task, 1 vCPU / 4 GB
  cpu  1    x 0.04048  = 0.04048 /hr
  mem  4    x 0.004445 = 0.01778 /hr
                        ---------
                         0.05826 /hr  x 730 = $42.53 / month

second task, 0.25 vCPU / 0.5 GB (the smallest thing you can buy)
  cpu  0.25 x 0.04048  = 0.01012 /hr
  mem  0.5  x 0.004445 = 0.00222 /hr
                        ---------
                         0.01234 /hr  x 730 =  $9.01 / month
```

So splitting the frontend into its own task costs **$9.01/month for nothing you
need**, plus ~$0.50/month for the Cloud Map hosted zone it then requires. Whereas
adding the frontend container *inside* the existing task costs **$0** in Fargate
charges - it consumes some of the 1 vCPU / 4 GB you already bought.

That is the whole cost argument, and it comes directly from "the task is the
billing unit."

---

## 4) Containers vs tasks, stated precisely (point 6, first half)

| | Container | Task |
| :-- | :-- | :-- |
| Is | one process tree, one image | a co-scheduled bundle of containers |
| Network | shares the task's namespace | owns one ENI, one IP, one port space |
| Lifecycle | can be restarted within a task | replaced as a whole |
| Billing | none of its own | **the unit of billing** |
| Ordering | expressible via `dependsOn` | not expressible between tasks |
| Failure | `essential: true` container dying kills the task | task dying triggers the service |

The `essential` flag is worth knowing: if a container marked `essential: true`
exits, ECS stops the entire task. This is how you say "the app is the point; if it
dies, tear the bundle down and let the service rebuild it."

---

## 5) "ECS cannot express cross-service ordering" (point 6, second half)

Your `docker-compose.yml` says:

```yaml
depends_on:
  backend:
    condition: service_healthy      # wait until backend's healthcheck passes
```

Compose can do this because a single daemon on a single host controls both
containers and can serialise their startup. It is a local, centralised decision.

ECS **services** are independent controllers. Each one's job is "keep N tasks of
revision R running." There is no field anywhere in the ECS API that says "do not
start service B until service A is healthy." Nothing coordinates two services'
startup, ever.

Why not? Because it would be a lie in a distributed system. Services are not
started once - they are *continuously reconciled*. Tasks are replaced at arbitrary
times by deployments, health failures, capacity events, and Spot reclamation. A
one-time ordering guarantee at creation is worthless when the backend can be
replaced at 3am while the frontend keeps running. So the orchestrator declines to
offer a guarantee it could not maintain.

**The design conclusion this forces, and it is a genuinely important one:**

> In a distributed system, you do not order startup. You make every component
> tolerate its dependencies being absent, and retry.

The frontend must handle "backend not reachable yet" as a normal, expected state -
show a spinner or a friendly error, and recover when the backend appears. This is
the same discipline as connection-retry loops around a database.

**However** - and this is the practical payoff - *within a single task* ECS **can**
express ordering, because one agent on one host controls those containers, exactly
like Compose:

```json
"dependsOn": [{ "containerName": "backend", "condition": "HEALTHY" }]
```

Conditions are `START`, `COMPLETE`, `SUCCESS`, `HEALTHY`. The `HEALTHY` condition
requires the depended-on container to declare a `healthCheck` in the task
definition.

So the recommended one-task design does not merely save $9/month. **It preserves a
correctness property your compose file relies on and that the two-service design
would have silently discarded.** That is the kind of thing worth noticing: the
cheaper architecture is also the more faithful one.

---

## 6) Fargate, properly (point 8)

ECS has two **launch types** - two answers to "whose computer does this run on."

**EC2 launch type.** You run a fleet of EC2 instances with the ECS agent installed.
The scheduler bin-packs tasks onto them. You own: capacity planning, AMI updates,
OS patching, scaling the fleet, and paying for idle instance capacity.

**Fargate launch type.** You declare CPU and memory; AWS runs the task on
infrastructure you never see. No instances in your account, nothing to patch,
nothing idle.

### What Fargate actually is under the hood

This is worth knowing because it explains Fargate's quirks. Fargate does not run
your container in a shared-kernel namespace next to other customers' containers.
Each task gets a **micro-VM** built with **Firecracker**, a minimal VMM (AWS
open-sourced it) that boots a stripped-down kernel in ~125 ms with only virtio
devices.

So Fargate is: **the container programming model, with virtual-machine isolation.**
Recall the security row of the container-vs-VM table in
[[S02a - Foundations - Processes, Namespaces, Containers]] - AWS does not trust
namespace isolation alone for multi-tenant compute, and this is how they resolve it.

That single design fact explains the quirks:

| Quirk | Why |
| :-- | :-- |
| Task start takes tens of seconds, not ms | a VM boots, and the image must be pulled |
| No `--privileged`, no arbitrary kernel modules | you do not own the kernel |
| No bind mounts, no Docker named volumes | there is no host you can address |
| Ephemeral storage 20 GB (to 200 GB), dies with the task | it is the micro-VM's disk |
| Docker `HEALTHCHECK` in the image is **ignored** | the agent, not Docker, supervises |

That last row is a real trap for FinSights. Both your Dockerfiles carry a
`HEALTHCHECK` instruction, and **Fargate will not run it.** The check must be
restated in the task definition's `healthCheck` block, or nothing is checked - and
then `dependsOn: HEALTHY` can never be satisfied.

### Fargate Spot

Fargate Spot uses AWS's spare capacity at roughly **70% off**, in exchange for AWS
being allowed to reclaim the task with a **two-minute SIGTERM warning**. For a
stateless demo app this is close to free money: ~$42.53/month becomes ~$12.80. The
honest caveat is that "AWS may take it away mid-demo," so: excellent for a
portfolio link, wrong for anything being graded live in front of you.

---

## 7) The two roles, because they are constantly confused

A task definition names two IAM roles that do completely different jobs at
different times.

```
  ecsTaskExecutionRole   -> used by the ECS AGENT, before your code runs
                            pull the image from ECR, create the log stream
                            
  taskRole               -> used by YOUR CODE, at run time
                            call Bedrock, read S3, query S3 Vectors
```

Getting the execution role wrong means the task never starts. Getting the task role
wrong means the task starts fine and then fails on the first real request - which
is exactly the `s3vectors:QueryVectors` denial recorded in your
`ECS_DEPLOYMENT_GUIDE.md`. That error is a *task role* error, and the fact that the
task was running and serving traffic when it happened is the proof your ECS
deployment was genuinely live.

[[S02e - Credentials and Config - Injection, IMDS, Loaders]] explains the mechanism
by which the task role reaches your code.

---

## 8) The three wiring options, decided (point 2)

You asked why these were presented as three alternatives. They are the three
answers to one question - *how does the frontend address the backend?* - and each
sits at a different point on a cost/capability curve. Categorically:

```
  A. shared network namespace    -> no name resolution at all
  B. DNS-based discovery         -> a name resolves to current addresses
  C. reverse proxy               -> a stable endpoint routes per request
```

Concretely, with the numbers from §3:

| | A: one task | B: two services + Cloud Map | C: add an ALB |
| :-- | :-- | :-- | :-- |
| `BACKEND_URL` | `http://localhost:8000` | `http://backend.finsights.local:8000` | via the ALB |
| Backend | 1 vCPU / 4 GB - $42.53 | $42.53 | $42.53 |
| Frontend | in the same task - **$0** | own task - $9.01 | $9.01 |
| Discovery | none needed | hosted zone - $0.50 | - |
| Load balancer | - | - | **+$16.43** |
| **Monthly** | **~$42.50** | **~$52.00** | **~$68.00** |
| 8000 exposed? | never | yes, or another SG | no |
| Startup ordering | `dependsOn: HEALTHY` | **impossible** | impossible |
| Stable public URL | no (task IP) | no | **yes, + HTTPS** |
| Independent scaling | no | yes, but unusable* | yes, real |

*Unusable because, per S02b §7, DNS discovery does not distribute load, and your
backend is `--workers 1`. B's headline benefit needs C to function.

**Decision: A.** It is cheapest, it keeps the metered API unexposed, and it is the
only one of the three that preserves your `depends_on: service_healthy` semantics.
B is strictly dominated - more expensive *and* less capable. C is a genuine future
upgrade, purely for the stable hostname and TLS, and it is additive: moving from A
to C later changes a task definition and adds a target group. It does not
invalidate anything you build now.

---

## 9) Carry-forward

1. Cluster (label) / task definition (versioned artifact) / task (running instance)
   / service (keeps N alive).
2. **Infrastructure that lives only in a console is not yours.** The task
   definition belongs in the repo.
3. The task exists because production processes come in co-scheduled bundles -
   Borg alloc -> k8s pod -> ECS task. Containers in one share a network namespace,
   hence `localhost`.
4. **The task is the billing unit**, with a 0.25 vCPU floor, so task *count* drives
   cost.
5. ECS cannot order two services because it reconciles continuously; components
   must tolerate absent dependencies and retry. Within one task, `dependsOn` works.
6. Fargate = container model + Firecracker micro-VM isolation. Hence no volumes, no
   privileged mode, slower starts, and **Docker `HEALTHCHECK` is ignored** - restate
   it in the task definition.
7. Execution role (agent, pre-start) is not the task role (your code, run time).
