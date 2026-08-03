
# S02e - Credentials and Config: Injection, IMDS, Loaders

Prerequisites: [[S02a - Foundations - Processes, Namespaces, Containers]] (a
container is a process born with a filesystem view),
[[S02c - AWS Substrate - EC2, VPC, Subnets, NAT, ECR]] (§5 on link-local ideas).

Answers points **3**, **4**, **5**, **12**.

This is the file where your reflection was closest to a genuine conceptual knot, so
it is worth reading twice.

---

## 1) The knot: "do you inject data into a machine, or a docker instance?"

Neither. The question contains a false premise, and dissolving it is most of the
learning.

From [[S02a - Foundations - Processes, Namespaces, Containers]]: a container is a
**process** that was born with a particular filesystem view and a particular set of
environment variables. There is no machine. There is no persistent instance sitting
there waiting to receive things.

So "injection" is not a verb that happens *to* a running container. It is a
description of **what was arranged before the process existed.** Nothing is pushed
in. The process opens its eyes already holding what it holds.

There are exactly **three** channels by which data reaches a containerised process,
and they differ in *when* they act:

```
  1. BUILD TIME   -> baked into an image layer          (permanent, shared, public-ish)
  2. START TIME   -> environment variables + mounts     (per-container, set at exec)
  3. RUN TIME     -> the process fetches it over the network / reads a file
```

Almost every confusion about secrets in containers is a confusion between these
three. Let me take them in order.

---

## 2) Channel 1 - build time, and why secrets must never go here

`COPY`, `RUN`, `ENV` in a Dockerfile write into an image layer. Layers are:

- **permanent** - `RUN rm secret.txt` in a later layer does *not* remove it. The
  earlier layer still contains it, and `docker history` will show it.
- **shared** - anyone who can pull the image has every byte of every layer.
- **content-addressed** - the layer digest is derived from its contents, so an
  identical secret produces an identifiable layer.

Hence the rule: **a secret in an image is a published secret.** Your project gets
this right, and deliberately. `ModelPipeline/.dockerignore` contains:

```
**/aws_credentials.env
**/.aws_secrets/
```

so the credentials file is not merely absent from the final image - it is never even
sent to the build daemon as part of the build context. That is the correct posture.

What *should* go at build time: code, dependencies, and non-secret defaults. Your
`backend.Dockerfile` sets `MODEL_PIPELINE_ROOT=/app` as an `ENV`, with a comment
calling it "CRITICAL - do not remove." That is a perfect build-time value: it is a
structural fact about the image's layout, identical in every environment, and not
secret.

---

## 3) Channel 2 - start time: what an environment variable actually is

Strip the mystique. When the kernel `exec`s a new program, it hands it three
things: `argv` (arguments), `envp` (an array of `KEY=VALUE` strings), and file
descriptors. The C runtime exposes `envp` as `environ`; Python exposes a copy as
`os.environ`.

That is the entire mechanism. An environment variable is **a string in an array
handed to a process at the moment of `exec`.**

Three consequences that matter and are rarely stated:

1. **They are set at birth and cannot be changed from outside afterward.** There is
   no `docker set-env` on a running container. To change one you replace the
   container. This is *why* task definitions are immutable and you register a new
   revision - the revision *is* the environment.
2. **They are inherited by children.** A subprocess gets a copy.
3. **They are readable by anyone who can read the process.** `/proc/<pid>/environ`,
   `docker inspect`, `ecs:DescribeTaskDefinition`. This is precisely why static AWS
   keys in a task definition are poor practice: anyone with read-only ECS
   permissions can retrieve them, and they never expire.

### `env_file` is a client-side convenience, not a container feature

This is a specific misconception worth killing. Your compose file says:

```yaml
env_file:
  - ../finrag_ml_tg1/.aws_secrets/aws_credentials.env
```

**The file does not go into the container.** The Docker *client*, running on your
Mac, reads that file, parses `KEY=VALUE` lines, and includes the resulting pairs in
the API call that creates the container. The container is then born with those
variables in its `envp`. It never sees a file, never knows a file was involved, and
the path is meaningless inside it.

So `env_file` is textual sugar over "type out a lot of `-e KEY=VALUE` flags." Once
you see that, the ECS question answers itself: **ECS has no `env_file` because
there is no client-side filesystem to read from.** The task definition's
`environment` array is the same destination, reached without the sugar.

### Mounts: the other start-time channel

A bind mount or volume (see S02a §7) adds an entry to the container's *mount
namespace* at creation. Again: arranged at birth, not injected later. The process
just finds a file at a path.

---

## 4) The 12-factor principle, and why it is more than a slogan

The relevant rule from the "twelve-factor app" methodology:

> **Store configuration in the environment**, where configuration is everything that
> varies between deployments.

The real test is: *could this value be different in staging than in production?* If
yes, it is config and belongs outside the image. If no, it can be baked in.

Applied to FinSights:

| Value | Varies by deploy? | Where it belongs |
| :-- | :-- | :-- |
| `MODEL_PIPELINE_ROOT=/app` | no - image layout | build time (`ENV`) |
| `BACKEND_URL` | **yes** - localhost vs DNS vs ALB | start time |
| `LOG_LEVEL` | yes | start time |
| AWS region | yes-ish | start time or config file |
| AWS credentials | yes, and secret | **neither - see §5** |
| the retrieval prompt templates | no | build time |

The payoff of the discipline is that **one image artifact runs in every
environment**. You do not build a "staging image" and a "prod image"; you run the
same bytes with different environment. That is what makes an image promotion
pipeline possible, and what makes "it worked in staging" mean something.

---

## 5) Why credentials are the exception, and what the good answer is

Environment variables are fine for config and mediocre for secrets:

- readable via `DescribeTaskDefinition` by anyone with read-only ECS access
- visible in `/proc/<pid>/environ` to anything in the container
- often echoed into logs by well-meaning debug code
- **they do not expire**, so a leak is permanent until you manually rotate

ECS offers two better mechanisms, and you should know they exist even though you
will not need them:

- `secrets` in the task definition, pulling from **Secrets Manager** or **SSM
  Parameter Store**. The ECS agent fetches the value at start and injects it as an
  env var. Better: the ciphertext lives in a managed store, access is IAM-audited,
  rotation is centralised. Still an env var in the end.
- **IAM roles.** No secret material anywhere, ever.

The second is strictly better and is what you should use. To understand *how* it can
possibly work, we need point **4**.

---

## 6) The link-local credential endpoint (point 4)

Your question, restated: *how can AWS "hand" a container credentials through a
local endpoint, when my code is written to read credentials from a file or
environment?*

The answer has two halves, and the first is the important one.

### Half one: your code does not do the looking. The SDK does.

You never wrote credential-fetching code. `boto3` did. When you call
`boto3.client('s3')` with no credential arguments, botocore walks a **credential
resolution chain**, in a fixed order, and stops at the first source that yields
credentials:

```
 1. explicit arguments to the client/session constructor
 2. environment variables    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
 3. shared credentials file  ~/.aws/credentials   (this is what AWS_PROFILE selects)
 4. shared config file       ~/.aws/config
 5. --> container credentials  (ECS / EKS): an HTTP GET to a link-local address
 6. --> instance metadata      (EC2 IMDS):   an HTTP GET to a link-local address
```

Steps 5 and 6 are the ones that feel like magic. They are just HTTP requests that
botocore knows how to make. **The abstraction is in the SDK, which is why your code
needs no change at all.** This is the single most useful thing to take from this
file: "use an IAM role instead of keys" is not an application change, it is a
*deployment* change, because the SDK already contains both code paths.

### Half two: the link-local address

`169.254.0.0/16` is the IPv4 **link-local** block. By definition it is never routed
- packets to it are not forwarded by any router, so an address in this range is
only meaningful on the local link. AWS uses this property to expose a per-instance,
per-task metadata service that is:

- reachable from inside the instance/task
- **unreachable from anywhere else in the world**, because the address cannot be
  routed to
- answered by the hypervisor or the local ECS agent, not by a server elsewhere

Two endpoints:

| Environment | Endpoint | How the SDK finds it |
| :-- | :-- | :-- |
| **EC2** | `169.254.169.254` (IMDS) | hardcoded, well-known |
| **ECS / Fargate** | `169.254.170.2` | via the env var `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` |

The ECS flow, concretely:

```
 1. You attach a task role to the task definition.
 2. ECS assumes that role via STS and holds temporary credentials for the task.
 3. ECS injects ONE environment variable into your container:
       AWS_CONTAINER_CREDENTIALS_RELATIVE_URI=/v2/credentials/<uuid>
 4. botocore sees that variable, GETs http://169.254.170.2 + that path.
 5. The agent returns JSON:
       { "AccessKeyId": "...", "SecretAccessKey": "...",
         "Token": "...", "Expiration": "2026-07-31T04:12:00Z" }
 6. botocore caches it and RE-FETCHES automatically before expiry.
```

```
   +---------------------------------------------------+
   |  Fargate task (micro-VM)                          |
   |                                                   |
   |   your process                                    |
   |     boto3 -> chain step 5                         |
   |        |  HTTP GET 169.254.170.2/v2/credentials/..|
   |        v                                          |
   |   [ ECS agent ]  <--- holds STS temp creds        |
   +--------|------------------------------------------+
            | assumed the task role via STS
            v
        IAM: taskRole  (policies say what you may do)
        
   nothing outside the task can reach 169.254.170.2 - it is not routable
```

So the answer to your question: it is not that AWS pushes a file to you, and not
that your code learns a new trick. It is that **AWS runs a tiny local HTTP server
that vends short-lived credentials, and the SDK you already use knows to ask it.**

Why this is strictly better than static keys:

- **Nothing secret is ever stored** - not in the image, not in the task definition,
  not in GitHub secrets, not on disk.
- Credentials **expire in hours** and rotate automatically. A leak has a short
  fuse.
- Permissions live in an IAM policy you can read, diff, and audit.
- You cannot accidentally commit them, because there is nothing to commit.

**One real caveat that applies to your code.** Nothing in FinSights reads
`AWS_SESSION_TOKEN`. Temporary credentials are a *triple* - key id, secret, and
session token - and a client given only the first two will be rejected. So your
static-key path (chain step 2, as your loader implements it) works **only with
long-lived IAM user keys**, never with temporary ones. That is another reason to
take the role path on ECS rather than trying to inject rotating keys as env vars.

---

## 7) Your MLConfig loader, analysed (point 5)

You asked whether it falls back. Here is the answer, from the code.

`ml_config_loader.py:56-61` is the decision point, and it is a **priority chain**,
not a fallback chain:

```python
# Priority 1: Check for AWS containerized environment (ECS, Lambda)
if os.getenv('AWS_EXECUTION_ENV') or os.getenv('AWS_LAMBDA_FUNCTION_NAME') \
   or os.getenv('ECS_CONTAINER_METADATA_URI'):
    print("[DEBUG] AWS containerized environment detected (ECS/Lambda) - using IAM role")
    self._aws_creds_source = "IAM_ROLE"
    return   # boto3 will automatically use the attached IAM role
```

It `return`s **before** ever constructing the `.aws_secrets/` path. Then every client
factory branches on that flag - `ml_config_loader.py:458-515`:

```python
def get_s3_client(self):
    if self._aws_creds_source == "IAM_ROLE":
        return boto3.client('s3', region_name=self.region)          # chain step 5
    return boto3.client('s3', aws_access_key_id=self.aws_access_key,
                        aws_secret_access_key=self.aws_secret_key,
                        region_name=self.region)                     # chain step 1
```

**So: yes, it handles the role path, and the answer to your practical question is
that moving to ECS requires zero code changes.** The Lambda work you thought you
wasted built exactly this, and it transfers intact - which is the concrete payoff of
point **12**'s pattern.

The order of priority is:

```
  1. AWS_EXECUTION_ENV / AWS_LAMBDA_FUNCTION_NAME / ECS_CONTAINER_METADATA_URI
                                            -> IAM_ROLE  (return immediately)
  2. .aws_secrets/aws_credentials.env exists -> load_dotenv, use those keys
  3. AWS_ACCESS_KEY_ID already in the env    -> use it
  4. none of the above                       -> FileNotFoundError
```

### The one genuine sharp edge

Step 4 raises rather than falling through to boto3's own chain. So a runtime that
*does* provide a role but does *not* set one of those three marker variables would
fail - plain EC2 with an instance profile, EKS with IRSA, App Runner. On
ECS/Fargate you are safe, because the agent sets both `AWS_EXECUTION_ENV=AWS_ECS_FARGATE`
and `ECS_CONTAINER_METADATA_URI_V4`.

Note the version detail: your code checks `ECS_CONTAINER_METADATA_URI` (v3), while
modern platforms set `ECS_CONTAINER_METADATA_URI_V4`. The `AWS_EXECUTION_ENV` check
covers you either way, but it is a fragile pair of conditions to rely on. **Cheap
insurance: set `AWS_EXECUTION_ENV=AWS_ECS_FARGATE` explicitly in the task
definition.** It is idempotent with what ECS injects and removes all doubt.

A second sharp edge, flagged honestly as **UNVERIFIED**: your Polars S3 reads do not
go through boto3 at all. `get_storage_options()` returns `{'aws_region': ...}` and
lets Rust's `object_store` resolve credentials itself. `object_store` does support
the ECS container-credentials endpoint, but I have not tested it in your stack.
It is the single most likely thing to break on cutover, and it would break on the
*first query*, not at startup.

---

## 8) The environment-recognition pattern (point 12)

Your instinct is right - this *is* a classic production pattern. It deserves a name
and a critique.

**The pattern.** A configuration object inspects its runtime environment at
construction and selects a strategy: where the root is, where credentials come
from, whether to stream from object storage or read local files. Your loader does
all three, and `MODEL_PIPELINE_ROOT` doubles as the switch that flips
`data_loading_mode` to `S3_STREAMING`.

**Why it is good.** It concentrates all environment-dependence in one auditable
place. The 400 modules downstream never ask "am I in the cloud?" - they receive a
configured object. That is genuine separation of concerns, and it is why your ECS
migration is a deployment change rather than a code change.

**The three ways it goes wrong**, all of which your code exhibits mildly, and all
worth knowing:

1. **Detection by proxy.** Checking `ECS_CONTAINER_METADATA_URI` to infer "I am in a
   container" couples you to a variable name AWS owns and has already versioned once
   (`_V4`). Detection is inherently guesswork about someone else's implementation.
2. **One variable, two meanings.** `MODEL_PIPELINE_ROOT` sets the filesystem root
   *and* silently selects S3 streaming. Two unrelated behaviours on one switch.
   Someone setting it for path reasons gets a data-loading change they did not ask
   for. There is no config key for the second effect, so it cannot be overridden
   independently.
3. **Raising instead of degrading.** Step 4 raising `FileNotFoundError` means a
   perfectly valid credential source (a plain instance profile) is rejected.

**The better version of the pattern**, worth knowing as the target shape:

> Prefer *explicit declaration* over *environment detection*. Let the deployment
> state what it is, with detection only as a convenience default.

```
  FINRAG_RUNTIME=ecs|lambda|local     <- the deployment declares itself
  FINRAG_CREDS=role|file|env          <- one variable, one meaning
  FINRAG_DATA_MODE=s3|local           <- independently overridable
```

Explicit configuration is boring, greppable, and cannot be surprised by a vendor
renaming an environment variable. Detection is a fallback for developer
convenience, not a foundation. This is the same instinct as "declare the task
definition in the repo" from
[[S02d - ECS Anatomy - Tasks, Services, Fargate]] - **prefer stated intent over
inferred state.**

---

## 9) Carry-forward

1. Nothing is injected into a running container. Three channels, all decided at or
   before `exec`: build time, start time, run time.
2. A secret in an image layer is published. `.dockerignore` is a security control.
3. An environment variable is a string in the array handed to `exec`. Set at birth,
   immutable from outside, readable by anyone who can read the process.
4. `env_file` is client-side sugar; the file never enters the container. This is why
   ECS has no equivalent.
5. Config in the environment means **one image runs everywhere**.
6. **`boto3` contains the credential chain, so "use a role" is a deployment change,
   not a code change.** Steps 5 and 6 are HTTP GETs to unroutable link-local
   addresses that vend short-lived credentials.
7. Your loader takes the `IAM_ROLE` branch and returns before touching the secrets
   file. ECS migration needs no code change. Set `AWS_EXECUTION_ENV` explicitly
   anyway, and smoke-test the Polars path.
8. Environment detection is a good pattern with three failure modes; explicit
   declaration is the better default.
