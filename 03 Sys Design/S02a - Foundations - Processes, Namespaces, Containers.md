
# S02a - Foundations: Processes, Namespaces, Containers

Prerequisite: [[S01 - Mechanisms 01]] ended at "Docker is a filesystem + process
isolation tool (namespaces/cgroups on Linux)." This file opens that sentence up.
Everything in the ECS/Fargate stack is a consequence of what is written here, so
read this one slowly and the later files get much easier.

Answers points **3 (partly)**, **7**, **12 (partly)**, and the header questions
*"what is a namespace"* and *"what is a logical grouping"*.

---

## 1) The process is the only unit that actually exists

Strip away every cloud word - container, task, service, cluster, function - and
what runs on a computer is a **process**: an instance of a program that the kernel
has given an address space, a thread of execution, and a table of open file
descriptors.

A process has exactly four things worth remembering:

- **An address space.** Virtual memory the kernel maps for it. Process A cannot
  read process B's memory; the MMU enforces this in hardware.
- **Threads.** One or more execution contexts sharing that address space.
- **File descriptors.** Small integers indexing kernel objects: open files,
  sockets, pipes. `0/1/2` are stdin/stdout/stderr by convention.
- **Credentials and a view of the world.** Its user id, its current directory,
  what it sees when it looks at the filesystem or the network.

That last one is the hinge. **A process's "view of the world" is not a property of
the world. It is a property of the process.** The kernel decides what each process
sees. Once that clicks, containers stop being mysterious.

```
        one physical machine, one kernel
        --------------------------------
  proc A        proc B        proc C
  addr space    addr space    addr space
  fds           fds           fds
  view --------> the kernel decides each view <-------- view
```

---

## 2) Namespaces: the kernel lying to a process, on purpose

**A namespace is a kernel-maintained scope for a class of system resource, such
that processes inside the namespace see only that scope's members and cannot see
or name anything outside it.**

That is the whole definition, and it is worth re-reading. A namespace is not a
folder, not a prefix, not a naming convention. It is an *enforced restriction on
what a process can perceive*.

Linux has several independent namespace types. The ones that matter here:

| Namespace | What it scopes | Effect when you get a fresh one |
| :-- | :-- | :-- |
| **PID** | process ids | Your process becomes PID 1 and cannot see host processes |
| **Mount (mnt)** | the filesystem tree | You get a different `/` entirely |
| **Network (net)** | interfaces, routes, ports, `/etc/resolv.conf` | You get your own `lo`, your own port space |
| **UTS** | hostname | You can be `finrag-backend` while the host is something else |
| **IPC** | shared memory, semaphores | Isolated from host IPC |
| **User** | uid/gid mapping | You can be root inside and nobody outside |

The important word is **independent**. These are separate dials. A process can
share the host's network namespace while having its own mount namespace. This is
not a curiosity - it is precisely the knob that ECS tasks turn, and it is the
mechanical reason the "one task, two containers, talk over localhost" design
works. Hold that thought; [[S02b - Networking - Sockets, DNS, Bridges, Discovery]]
collects on it.

### The intuition to keep

Ordinary process isolation says *"you cannot touch my memory."*
Namespaces say *"you cannot even name the thing you would touch."*

A process in a fresh PID namespace does not fail to kill PID 4471 on the host. It
has no way to *refer* to PID 4471. The number means something different inside.

### "Logical grouping" - the phrase demystified

When AWS documentation says an ECS cluster is a "logical grouping," it means the
opposite of a namespace. There is **no enforcement**. A cluster is a label on a
record in a database. Two tasks in different clusters are not isolated from each
other in any way - they can reach each other over the network if the network
permits it. The cluster boundary exists for your organisation, billing tags, and
IAM policy scoping, not for security.

Whenever a cloud doc says "logical," read it as: **an accounting fiction, not a
wall.** Whenever it says "namespace" in the Linux sense, read it as: **a wall.**
AWS also uses "namespace" in the Cloud Map / DNS sense, which is a *third* thing
again - closer to a DNS suffix. Same word, three meanings, and conflating them is
the single most common source of confusion here.

---

## 3) cgroups: the other half of the trick

Namespaces control **what a process can see**. Control groups (cgroups) control
**how much it can consume**: CPU shares, memory ceiling, block I/O, process count.

Namespaces without cgroups gives you a process that cannot see the host but can
eat all its RAM. cgroups without namespaces gives you a process that is politely
limited but can read everything. You need both.

```
namespaces  ->  perception boundary   ("what exists for me")
cgroups     ->  consumption boundary  ("how much I may take")
              container = both, applied to a process
```

The memory ceiling is a hard one: exceed the cgroup memory limit and the kernel's
OOM killer terminates the process. **It does not slow down. It dies.** This is
exactly the mechanism that would have killed the FinSights backend had it been
deployed at the documented 1024 MB while loading a 344.5 MB table per request with
Polars-to-Pandas conversions on top. Not "it would have been slow" - the kernel
would have shot it.

---

## 4) So: a container is a process, not a machine

**A container is one or more processes running on the host kernel, with their own
namespaces and cgroup limits.** That is all. There is no guest operating system,
no virtual CPU, no boot sequence.

This is why:

- Containers start in **milliseconds**. There is nothing to boot. The kernel
  creates namespaces and `exec`s your binary.
- `docker stats` shows a process's memory, not a VM's.
- A Linux container cannot run on a Windows kernel. Docker Desktop on your Mac
  runs a **Linux VM** and puts containers inside *that*. Your Mac is not running
  those containers natively; it is running one VM which runs them.
- **You cannot "log into" a container** the way you log into a machine. When you
  run `docker exec -it ... bash`, you are asking the daemon to start *a new
  process* inside the same namespaces. There is no login, no session, no sshd.

That last point is the correction to point **3** in your reflection, where you ask
whether you "inject data and keys into a machine, or a docker instance." Neither.
There is no machine and there is no persistent instance to inject into.
[[S02e - Credentials and Config - Injection, IMDS, Loaders]] does this properly,
but the foundation is here: **a container is a process that was born with a
particular filesystem view and a particular set of environment variables, and
both were decided at the instant of birth.**

### Container vs VM, honestly

| | Virtual machine | Container |
| :-- | :-- | :-- |
| Isolation by | hypervisor, virtual hardware | kernel namespaces + cgroups |
| Guest kernel | its own | none - uses the host's |
| Boot time | tens of seconds | milliseconds |
| Overhead | full OS in RAM | the process only |
| Security boundary | strong | weaker - one kernel, shared attack surface |
| Cross-OS | Linux on Windows fine | needs a matching kernel |

Note the security row. It is why AWS Fargate, despite presenting a container API,
runs each task inside a **lightweight micro-VM** (the Firecracker hypervisor).
AWS does not trust namespace isolation alone for multi-tenant workloads. You get
the container programming model with VM-grade separation, and you pay for it in
task startup time (tens of seconds, not milliseconds). This is directly relevant
to point **8**; [[S02d - ECS Anatomy - Tasks, Services, Fargate]] finishes it.

---

## 5) Images, layers, and where "the filesystem" comes from

A container needs a root filesystem. That comes from an **image**.

An image is an ordered stack of read-only **layers**, each a tarball of filesystem
changes, plus a JSON manifest of metadata (default command, environment variables,
exposed ports, entrypoint). Each `RUN` / `COPY` / `ADD` in a Dockerfile produces
one layer.

At run time the engine stacks the layers with a union filesystem (overlayfs) and
adds one thin **writable layer** on top:

```
   +-------------------------------+
   |  writable layer (per container, dies with it)
   +-------------------------------+
   |  layer 4: COPY app code       |  \
   +-------------------------------+   |
   |  layer 3: pip install deps    |   |  read-only,
   +-------------------------------+   |  shared between
   |  layer 2: apt-get ...         |   |  all containers
   +-------------------------------+   |  from this image
   |  layer 1: python:3.12-slim    |  /
   +-------------------------------+
```

Three consequences worth memorising:

1. **Writes go to the writable layer via copy-on-write.** Modify a file that lives
   in layer 2 and the whole file is copied up first. Large-file rewrites inside a
   container are surprisingly expensive.
2. **The writable layer is destroyed with the container.** Restart, and every
   change is gone. This is not a bug or a safety feature; it is the direct
   consequence of the layer being scoped to the container's lifetime.
3. **Layers are content-addressed and shared.** Ten containers from one image
   share one on-disk copy of the read-only layers. This is why a 900 MB image does
   not cost 9 GB for ten containers, and why layer caching makes rebuilds fast.

`image` is to `container` what a class is to an instance, or - closer to your
S01 note - what a `.pyc` on disk is to a running interpreter with state.

---

## 6) Statelessness: the word does not mean what people think

Point **7** asks about this directly, and the standard explanations are bad.

"Stateless" does **not** mean the process holds no variables in memory. Every
process holds state in memory. It means something much more specific:

> **A service is stateless when no state that must survive a request is stored
> only inside the serving process or its local disk.**

The test is not "is there state?" It is: **if I destroy this container right now
and start a fresh one, is anything lost that mattered?**

Three grades:

- **Stateless.** All durable state is external - a database, S3, a cache
  service. Kill any instance at any time; you lose at most in-flight requests.
- **Stateful with local durability.** State is on the container's disk and must
  survive. Needs volumes, and needs care about *which* instance restarts where.
- **Session-affine.** Durable state is external, but a *client* is bound to a
  particular process for the life of its session. Kill that process and the user
  is disrupted even though nothing is permanently lost.

That third grade is the one nobody teaches, and it is exactly what Streamlit is.
Point **10** in your notes spots this instinctively. Held for
[[S02f - Serving and Production - Processes, Health, Cost]].

### Why FinSights is genuinely stateless, and why that is lucky

Your `docker-compose.yml` has **no `volumes:` key and no bind mounts anywhere**.
Every durable thing lives in AWS:

- fact and dimension tables, embeddings meta -> S3 (`sentence-data-ingestion-mjs`)
- the vector index -> S3 Vectors (`finrag-embeddings-s3vectors`)
- query logs -> S3
- model weights -> not yours at all; Bedrock holds them

The only local writes are the `/tmp/finrag_cache` parquet cache, which is a pure
performance cache - if it vanishes, the next read re-downloads from S3 and the
answer is identical.

**This is the single property that makes the whole cloud story easy.** Fargate,
scale-to-zero, destroy-and-rebuild, task replacement - all of it is available to
you *because* there is nothing to preserve. Most real deployments spend most of
their complexity budget on state. You have none of that cost. Point **19**'s
instinct - "I don't mind tearing down and waiting, as proof of reliable
zero-to-reproduction build" - is only a sane thing to want because of this.

---

## 7) Volumes and bind mounts

Both solve "the writable layer dies." They differ in *who owns the storage*.

**Bind mount** - a host path grafted into the container's mount namespace:

```yaml
volumes:
  - ../finrag_ml_tg1/data_store/logs:/app/finrag_ml_tg1/data_store/logs
     ^ host path (real, you can ls it)     ^ path inside the container
```

The container writes to what it thinks is `/app/.../logs`; the kernel redirects to
the host directory. Exactly the same file, two names, because *a mount namespace
is a mapping and you just added an entry to it*. Your compose file has this,
commented out, at lines 34-36.

Bind mounts are a **development** tool. They are how you edit code on your Mac and
see it change inside a running container without rebuilding. They are also
non-portable by nature: they hardcode a path on one machine.

**Named volume** - storage the engine creates and manages, identified by name:

```yaml
volumes:
  - finrag-cache:/app/cache
```

You do not know or care where it lives on the host. It survives `docker compose
down`, is removed by `docker compose down -v` (this is what the `-v` means), and
is listed by `docker volume ls`.

**The relevant fact for your project:** you ran `docker volume ls` and
`docker system df` and both showed zero volumes. `down -v` is a no-op on your
stack. And **Fargate supports neither bind mounts nor Docker named volumes** - it
offers ephemeral task storage (20 GB by default, configurable to 200 GB, gone when
the task stops) and optional EFS mounts for real persistence. Since you need
neither, an entire category of cloud difficulty simply does not apply.

```
writable layer   -> dies with the container      (default)
bind mount       -> lives on a specific host     (dev convenience, not portable)
named volume     -> lives in engine storage      (portable-ish, survives restarts)
Fargate ephemeral-> dies with the task           (what you actually get)
EFS mount        -> lives in a network filesystem(real persistence, costs money)
```

---

## 8) Carry-forward

Six sentences to hold as you move to the next file:

1. A process's view of the world is a property of the process; the kernel decides
   it.
2. A namespace is an enforced perception boundary. A cgroup is an enforced
   consumption boundary. A container is a process with both.
3. "Logical grouping" in cloud docs means an accounting label with no enforcement.
   Do not confuse it with a namespace.
4. An image is read-only layers plus metadata; the container adds one writable
   layer that dies with it.
5. Stateless means "destroying this instance loses nothing that mattered" - and
   FinSights genuinely is, which is why every cheap cloud option is open to you.
6. **Network namespaces are an independent dial from mount namespaces.** Two
   processes can share one network view while having different filesystems.

That sixth point is the seed of the next file, and of the answer to point **2**.
