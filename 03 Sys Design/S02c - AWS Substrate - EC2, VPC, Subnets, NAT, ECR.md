
# S02c - AWS Substrate: EC2, VPC, Subnets, NAT, ECR

Prerequisites: [[S02a - Foundations - Processes, Namespaces, Containers]],
[[S02b - Networking - Sockets, DNS, Bridges, Discovery]].

Answers points **9**, **14**, and the header questions *EC2*, *ECR*,
*virtual cloud machines*, *VPCs*, *CloudWatch log groups*, *Cloud Map namespaces*.

The organising idea: **AWS did not invent new concepts. It rented you the ones from
the previous two files, at datacentre scale, with a billing meter attached.** Read
each service below as "which primitive is this, and what does the meter charge for."

---

## 1) EC2, and what "virtual cloud machine" actually means

EC2 (Elastic Compute Cloud) rents **virtual machines**. A physical server in an AWS
datacentre runs a **hypervisor**; the hypervisor slices the real CPU, RAM, and
devices into virtual ones and presents each slice to a guest OS that believes it
owns a computer.

This is the *other* isolation technology from [[S02a - Foundations - Processes, Namespaces, Containers]] -
not namespaces sharing one kernel, but virtual hardware, each guest with its own
kernel. Stronger isolation, higher overhead, slower to start.

An **instance type** encodes the slice: `t3.medium`, `m5.large`, `c7g.xlarge`. The
letter is the family (t = burstable, m = general, c = compute-optimised, r =
memory-optimised, g suffix = ARM/Graviton), and the size is the slice.

Two things worth knowing because they contradict common intuition:

- **Modern AWS hypervisor work happens on dedicated hardware.** The Nitro system
  offloads networking and storage virtualisation to purpose-built cards, so guests
  get near-bare-metal performance. "Virtualisation is slow" is a decade out of date.
- **You are billed per second while the instance exists, whether or not it is
  doing anything.** A VM's cost is a function of *time held*, not *work done*.
  This is the mental model to break out of; the whole appeal of Fargate and
  scale-to-zero is escaping it.

You did not use EC2 directly for FinSights, and that was correct. What matters is
knowing what you avoided: OS patching, capacity planning, AMI management, and
paying for idle CPU.

### The geography: region and availability zone

A **region** (`us-east-1`) is a geographic area. An **availability zone**
(`us-east-1a`) is one or more discrete datacentres inside it, with independent
power and cooling. AZs in a region are linked by low-latency private fibre.

The design rule: **anything you want to survive a datacentre failure must exist in
at least two AZs.** For FinSights this is mostly free - S3 and Bedrock are already
regional and replicated, and a subnet exists per AZ so you simply list several when
launching a task.

Everything about FinSights is in `us-east-1`, which is also where S3 Vectors and
your Bedrock models live. Keeping compute in the same region as data is not
cosmetic: cross-region data transfer costs money and adds latency.

---

## 2) The VPC is a network namespace you rent

A **VPC** (Virtual Private Cloud) is a software-defined private network inside AWS,
identified by a CIDR block such as `172.31.0.0/16`.

Read it exactly as the cloud-scale version of the bridge network from
[[S02b - Networking - Sockets, DNS, Bridges, Discovery]]. Same job: a private
address space in which your things can address each other, isolated from everyone
else's. Different scale: spans a whole region, implemented in AWS's network
fabric rather than one host's `iptables`.

CIDR notation, since it recurs: `172.31.0.0/16` means the first 16 bits are the
network prefix and the remaining 16 bits are host addresses - so
`172.31.0.0`-`172.31.255.255`, about 65,000 addresses. A `/24` leaves 8 bits, so
256 addresses. **Higher number after the slash = smaller network.**

Your account has exactly one VPC: `vpc-07e7c1c0f47896c94`, `172.31.0.0/16`, flagged
`IsDefault=true`. AWS creates one per region automatically with a subnet in each
AZ, all internet-reachable and ready to use. **Your old deployment used it, and
that was the right call** - see section 5.

---

## 3) Subnets, and the real definition of "public"

A **subnet** is a slice of the VPC's CIDR bound to exactly one availability zone.
Every network interface lives in exactly one subnet.

Now the part almost everyone gets wrong, and which point **9** and point **14** both
touch:

> **A subnet is not public or private because of a setting on the subnet. It is
> public or private because of what its route table says.**

There is no `isPublic` flag. The distinction is entirely derived:

- A **public subnet** has a route table with a default route (`0.0.0.0/0`) pointing
  at an **internet gateway**.
- A **private subnet** does not. Its default route points at a NAT gateway, or
  nowhere at all.

### Route tables and the internet gateway

A **route table** is a list of "destination CIDR -> target" rules attached to a
subnet. The kernel-level analogue is your machine's routing table; same idea.

```
Destination        Target
172.31.0.0/16      local            <- traffic inside the VPC stays inside
0.0.0.0/0          igw-xxxxxxxx     <- everything else goes to the internet gateway
```

An **internet gateway (IGW)** is a VPC-attached component that routes between the
VPC and the public internet, and performs 1:1 NAT between a resource's private
address and its public address. Two properties matter enormously:

- **An IGW is free.** No hourly charge, no per-GB charge for the gateway itself.
- **An IGW is bidirectional but requires a public IP on the resource.** A task
  with only a private address cannot use it, even in a public subnet.

That second point is the entire reason NAT gateways exist.

---

## 4) The NAT gateway, and why it is the classic AWS money trap

The problem it solves: a resource in a **private** subnet has no public IP, so it
cannot use the IGW - but it still needs *outbound* internet access, to call an API,
download a package, or reach S3.

A **NAT gateway** is a managed device that sits in a *public* subnet, accepts
traffic from private subnets, rewrites the source address to its own public
address, and forwards it out. Return traffic comes back through it. The result is
outbound-only internet: private resources can initiate connections outward, and
nothing on the internet can initiate a connection inward.

That asymmetry is genuinely valuable for security. It is also expensive:

| | Charge |
| :-- | :-- |
| NAT gateway, hourly | ~$0.045/hour = **~$32.85/month**, per gateway, per AZ |
| NAT gateway, data processing | ~$0.045 per GB **in addition to** normal transfer |
| Internet gateway | **$0.00** |

Read those two rows together and the shape of point **14** becomes clear.
**One NAT gateway costs more per month than the entire FinSights backend
container.** And the properly-architected version wants one per availability zone
for redundancy, so the textbook diagram is ~$66/month before a single byte moves.

### What your deployment did instead, and why it was right

The old `setup-infrastructure.yml` never created a VPC, never created a private
subnet, and never created a NAT gateway. It discovered the default VPC, enumerated
its (all-public) subnets, and launched tasks with `assignPublicIp=ENABLED`.

So each task got its own public address and used the **free** internet gateway
directly to reach Bedrock and S3.

```
   TEXTBOOK                              WHAT YOU DID
   --------                              ------------
   private subnet                        public subnet
     task (no public IP)                   task (public IP)
        |                                     |
     NAT gateway  ~$33/mo                  internet gateway   $0
        |                                     |
     internet gateway  $0                  internet
        |
     internet
```

The tradeoff is real and worth stating honestly: your tasks are *addressable from
the internet*, so their protection rests entirely on the security group rather than
on being unroutable. For a two-container demo app with no inbound attack surface
you care about, that is a sound trade. For a system holding customer data it would
not be.

**The general lesson, which is the most valuable thing in this file:** in small AWS
systems the fixed-price managed components dominate the bill. NAT gateway ~$33,
ALB ~$16, versus ~$42 for the compute doing the actual work. Architectural
sophistication has a monthly price, and on a student project the sophisticated
choice is often the wrong one.

---

## 5) Security groups: a stateful firewall per interface

A **security group** is a firewall attached to network interfaces. Rules are
**allow-only** (there is no deny rule) and **stateful** - if you allow an inbound
connection, the return traffic is automatically permitted, so you almost never need
matching egress rules.

Default posture: **all inbound denied, all outbound allowed.**

Your old group, `finsights-backend-sg`, had exactly one inbound rule:

```
tcp/8000 from 0.0.0.0/0
```

Two observations, and the second corrects something I said earlier.

First, **port 8501 was never opened**, yet the deploy workflow assigned this same
group to the frontend and then printed `http://<ip>:8501`. Since you did present a
working public site, that group must have been edited by hand in the console - a
change that lives nowhere in the repository. This is the recurring theme of the
whole ECS post-mortem: the working configuration existed only in the console.

Second: I earlier framed "8000 open to the world" as a security hole, and your
point **18** pushes back - the site was *meant* to be public, and you shipped it
publicly, on purpose. You are right and I want to be precise about what remains
true. There are two different doors:

- **The frontend on 8501 being public was the product.** Correct by design.
- **The backend API on 8000 being public was incidental.** Anyone who found the IP
  could `POST /query` directly, bypassing your UI, and each call spends real
  Bedrock money at ~$0.02-0.03 a query.

So the concern was never "the site is public." It was "the metered API is
*separately* public with no rate limit." Co-locating the containers closes that
door for free by never opening 8000 at all - the frontend stays as public as you
want it, and the paid endpoint stops being independently reachable. Same public
product, one fewer unmetered door.

A note on **NACLs**, which you will meet in reading: network ACLs are a second,
subnet-level firewall that *is* stateless and *does* support deny rules. You almost
never need them. Default NACLs allow everything; leave them alone.

---

## 6) ECR: a registry, nothing more

**ECR** (Elastic Container Registry) stores container images. It is the same kind of
thing as Docker Hub: a content-addressed store of layers plus manifests, with an
HTTP API and IAM-based authentication.

What you need to know:

- Images are addressed as
  `<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`. The account number is
  in the hostname, which is exactly why the old workflow's hardcoded
  `729472661729.dkr.ecr.us-east-1.amazonaws.com` is unusable in your current
  account (`908877262866`).
- **Authentication is a 12-hour token**, obtained via `aws ecr get-login-password`
  and fed to `docker login`. There is no permanent registry password.
- **Storage is $0.10 per GB-month** and it stores *compressed* layers, so your
  ~1.5 GB of local images bill as roughly 0.6 GB, about **six cents a month**. This
  is the entire cost of a fully torn-down-but-still-rebuildable deployment.
- **Tags are mutable by default.** Pushing `:latest` twice silently replaces it.
  This is why the old pipeline's habit of pushing an immutable `:${git-sha}` tag
  and then *deploying `:latest` anyway* destroyed its own ability to roll back.

A registry is the handoff point between build and run: **build once, push, and then
any number of runtimes pull the identical bytes.** That property is what makes
"works on my machine" go away, and it is why the image - not the Dockerfile - is
the real deployment artifact.

---

## 7) CloudWatch log groups

Containers write to stdout/stderr. Something must collect that or it vanishes with
the writable layer.

CloudWatch Logs has a three-level hierarchy:

```
log group      /ecs/finsights          <- the container/app, retention set here
  log stream   backend/abc123...       <- one per container instance
    events     timestamped lines
```

Mechanically, the ECS agent takes the container's stdout via the `awslogs` log
driver, configured in the **task definition**, and ships each line as an event.

Four practical facts:

1. **Retention defaults to "never expire."** Storage bills forever until you set
   `retentionInDays`. Setting 14 or 30 days is the single highest-value
   observability hygiene action available.
2. **The log group must exist before the task starts,** or the task definition must
   set `awslogs-create-group: "true"`. Otherwise the task fails to start with
   `ResourceInitializationError` - and the standard
   `AmazonECSTaskExecutionRolePolicy` grants `CreateLogStream` and `PutLogEvents`
   but **not** `logs:CreateLogGroup`. The old FinSights setup created no log group
   at all, which is a latent version of exactly this failure.
3. **Ingestion is the real cost** (~$0.50/GB), not storage (~$0.03/GB-month). Chatty
   debug logging is what makes CloudWatch expensive.
4. **Log Insights** is the query language over these events. It is how you answer
   "show me every 5xx in the last hour," and it is the thing the misfiled
   `AWS_LogMonitoring_Analytics.md` should have contained and did not.

---

## 8) Cloud Map namespaces - the third meaning of "namespace"

[[S02a - Foundations - Processes, Namespaces, Containers]] warned that "namespace"
has three unrelated meanings in this stack. Here is the third.

**AWS Cloud Map** is managed service discovery. A **namespace** in Cloud Map is a
DNS domain that groups service names - `finsights.local`. Underneath, a private
namespace is literally a **Route 53 private hosted zone** visible only inside your
VPC.

The flow: you create namespace `finsights.local`, register a service `backend`
inside it, and attach it to an ECS service via `--service-registries`. ECS then
registers and deregisters each task's IP as tasks start and stop. Clients resolve
`backend.finsights.local` and get current healthy addresses.

This is precisely the **DNS-based discovery pattern** from
[[S02b - Networking - Sockets, DNS, Bridges, Discovery]] - the managed equivalent
of Docker's `127.0.0.11` embedded resolver.

Two corrections to your old documentation:

- `INFRASTRUCTURE_SETUP_GUIDE.md` claims *"Service Discovery: Free."* It is not.
  The underlying private hosted zone bills about **$0.50/month**, plus small
  per-resource and per-query charges.
- And per section 7 of the networking file, it solves *churn* and not
  *distribution*. It is not a load balancer.

Note that this namespace is a **naming scope with no enforcement** - closer to the
"logical grouping" sense than the Linux sense. Nothing stops a task outside the
namespace from talking to one inside it.

---

## 9) The guard-and-check create pattern

Point **9** asks about "guard and check creates." This is a real and important
infrastructure-scripting idea, and your old workflow used it consistently - it is
the best thing in those files.

The problem: infrastructure scripts get re-run. A script that blindly does
`create-cluster` fails the second time with "already exists," and now your pipeline
is red for a reason that is not a problem.

The property you want is **idempotency**: *running the operation N times leaves the
same state as running it once.* The guard pattern achieves it by checking first:

```bash
if aws ecr describe-repositories --repository-names "$REPO" 2>/dev/null; then
  echo "exists, reusing"
else
  aws ecr create-repository --repository-name "$REPO" ...
fi
```

This is what "declarative" tools like Terraform and CloudFormation do for you
automatically: you declare desired state, they diff against actual state and
reconcile. Hand-rolled CLI scripts are **imperative**, so you must build
idempotency yourself.

### The trap your old workflow fell into, which is worth internalising

Guarding on *existence* while performing *multiple* operations inside the `else`
branch is not idempotent - it is only *first-run correct*.

```bash
if aws iam get-role --role-name "$ROLE" 2>/dev/null; then
  echo "exists"                       # <-- does NOT verify policies
else
  aws iam create-role ...             # step 1
  aws iam attach-role-policy ...      # step 2  <- if THIS fails...
fi
```

If step 1 succeeds and step 2 fails, the re-run sees the role exists, takes the
`then` branch, and **never repairs the missing policy**. You now have a role that
exists and can do nothing, and a green pipeline.

The fix is a rule worth memorising: **guard only the non-idempotent operation, and
let naturally-idempotent operations run unconditionally.** `attach-role-policy` and
`put-role-policy` are both safe to repeat, so they belong *outside* the guard:

```bash
if ! aws iam get-role --role-name "$ROLE" 2>/dev/null; then
  aws iam create-role ...             # guarded: not idempotent
fi
aws iam attach-role-policy ...        # unguarded: idempotent, self-healing
aws iam put-role-policy ...           # unguarded: idempotent, self-healing
```

Now every run reconciles policy state. That is the difference between a script that
*creates* infrastructure and a script that *converges* infrastructure - and it is
the property you want for point **19**'s zero-to-reproduction goal.

---

## 10) Carry-forward

1. EC2 rents virtual machines via a hypervisor - the *other* isolation model, and
   billed for time held rather than work done.
2. A VPC is a rented network namespace; a subnet is an AZ-bound slice of it.
3. **Public vs private is a property of the route table, not the subnet.**
4. An internet gateway is free. A NAT gateway is ~$33/month and exists only to give
   outbound access to resources with no public IP. Skipping it was correct.
5. Security groups are stateful allow-only firewalls. The public *frontend* was the
   product; the separately public *paid API* was the accident.
6. ECR is a registry; the image is the deployment artifact; ~6 cents/month.
7. CloudWatch log groups need retention set and must exist before the task starts.
8. Cloud Map = managed DNS discovery = a Route 53 private hosted zone, ~$0.50/mo,
   and not free as your docs claim.
9. Guard the non-idempotent operation only; let idempotent operations self-heal.
