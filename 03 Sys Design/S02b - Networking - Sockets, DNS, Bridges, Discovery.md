
# S02b - Networking: Sockets, DNS, Bridges, Service Discovery

Prerequisite: [[S02a - Foundations - Processes, Namespaces, Containers]], in
particular that a **network namespace** is an independent dial.

Answers points **1**, **2 (the mechanism half)**, and the header questions
*"embedded DNS"*, *"network bridges"*, *"service discovery"*.

---

## 1) The four-tuple: what "a connection" is

Before DNS, before containers, get this exact.

A TCP connection is identified by four numbers:

```
( source IP , source port , destination IP , destination port )
```

That tuple is the connection's identity. The kernel keeps a table of them. When a
packet arrives, the kernel matches the tuple and hands the payload to the process
holding the matching socket.

Two roles:

- A **listening socket** is bound to `(local IP, local port)` and does nothing but
  wait. `uvicorn --host 0.0.0.0 --port 8000` creates one. `0.0.0.0` is not an
  address - it means *"every local interface"*. `127.0.0.1` would mean *"only the
  loopback interface"*, i.e. only connections originating on this machine.
- A **connected socket** carries a full four-tuple and moves bytes.

**Ports are per-network-namespace, not per-machine.** This is the fact that
unlocks everything else in this file. Two processes cannot both listen on
`0.0.0.0:8000` in the *same* network namespace - the second gets `EADDRINUSE`.
Put them in *different* network namespaces and both succeed, with no conflict,
because there are now two independent port spaces.

That is not a workaround. That is the design.

---

## 2) `localhost` is a namespace-scoped concept

`localhost` resolves to `127.0.0.1`, the loopback interface. Loopback is a virtual
interface the kernel provides that never touches physical hardware - packets sent
to it are handed straight back to the same network namespace.

Therefore:

> **`localhost` does not mean "this machine." It means "this network namespace."**

This one sentence is why point **2**'s first option exists. If two processes share
a network namespace, they share a loopback interface and a port space, so one can
reach the other at `localhost:8000` with no DNS, no service discovery, no
configuration whatsoever. It is the cheapest possible inter-process link that
still uses the network stack, and it cannot fail for name-resolution reasons
because no name is resolved.

When your Streamlit frontend's default is `http://localhost:8000`
(`frontend/config.py:21`), that default is *correct* in exactly one topology: the
two processes sharing a network namespace. In Docker Compose they do not, which is
why compose has to override it. In a single ECS task they do - which is why the
recommended design gets to delete the override.

---

## 3) DNS, from the top

DNS turns a name into an address. The mechanics you need:

A **resolver** is library code inside your process (`getaddrinfo`). It reads
`/etc/nsswitch.conf` and `/etc/resolv.conf` to learn where to ask, then sends a
UDP query to a **nameserver** on port 53 and gets back records.

Record types worth knowing:

| Type | Maps | Note |
| :-- | :-- | :-- |
| **A** | name -> IPv4 | the common case |
| **AAAA** | name -> IPv6 | |
| **CNAME** | name -> another name | one alias, resolved again |
| **SRV** | name -> host **plus port** and priority/weight | the "real" service-discovery record |

Two properties that cause real production bugs:

**TTL.** Every record carries a time-to-live. Resolvers cache for that long. A
10-second TTL means a changed address propagates in ~10 seconds - which is why the
old FinSights Cloud Map service used `TTL=10`. Short TTL means fast failover and
more query volume.

**Caching happens at several layers, and one of them is your own process.** This
is the killer. Python's `socket.getaddrinfo` does not cache, but a `requests`
Session with a pooled connection does not even re-resolve - it reuses the open TCP
connection. So a name that now points somewhere else is irrelevant to a client
that already holds a connection. Come back to this in section 7; it is the reason
Cloud Map alone cannot load-balance your app.

```
process ------> resolver (libc)  ------> nameserver ------> authoritative data
   |               |                        |
   |               reads /etc/resolv.conf   caches per TTL
   |
   holds an open connection ---> may never re-resolve at all
```

---

## 4) Bridges and veth pairs: how container networking actually works

Now the mechanism behind point **1**.

Give a container its own network namespace and it starts with nothing but a
loopback interface. It cannot reach anything. To connect it, the engine builds two
things.

**A veth pair.** A virtual Ethernet cable: two linked interfaces where bytes into
one come out of the other. One end is placed in the container's namespace (it
appears as `eth0` inside); the other stays in the host's namespace.

**A bridge.** A software Layer-2 switch in the host namespace (`docker0`, or a
per-network bridge). Every container's host-side veth end is plugged into it. The
bridge learns MAC addresses and forwards frames between its ports, exactly like a
physical switch.

```
       container A ns              container B ns
       +--------------+            +--------------+
       | eth0         |            | eth0         |
       | 172.18.0.2   |            | 172.18.0.3   |
       +------|-------+            +------|-------+
              | veth pair                 | veth pair
       -------|---------------------------|--------  host network namespace
              |                           |
          +---+---------------------------+---+
          |        bridge: finrag-network      |   <- software L2 switch
          +------------------+-----------------+
                             | NAT / iptables masquerade
                             v
                        host eth0 -> the internet
```

So when your compose file says:

```yaml
networks:
  finrag-network:
    driver: bridge
```

it is asking for one software switch, and both containers get a veth into it. They
are on the same Layer-2 segment with addresses from the same private subnet, so
they can reach each other directly - **but by IP, which nobody wants to hardcode.**
Hence DNS.

### Port publishing is NAT, not magic

`ports: - "8000:8000"` does not move the listener. The process still listens on
`0.0.0.0:8000` inside its own namespace. Docker installs a **DNAT rule** in the
host's iptables: traffic arriving at host port 8000 gets its destination rewritten
to `172.18.0.2:8000` and forwarded over the bridge.

This is why the note at the end of your `LOC_DOCKER_README.md` is right that a
container cannot see the host's LAN IP. The container's network namespace contains
only its veth and loopback. `192.168.1.100` is not an address that exists in its
world, so Streamlit's "Network URL" auto-detection has nothing to detect.

---

## 5) Embedded DNS: the answer to point 1

On a user-defined bridge network, the Docker engine runs a small DNS server and
points each container's `/etc/resolv.conf` at **`127.0.0.11`**.

Look closely at that address. It is in the `127.0.0.0/8` loopback block - so it is
only reachable *from inside that container's network namespace*. The engine
intercepts traffic to it and answers from its own table of container names,
network aliases, and service names. Queries for anything it does not know are
forwarded upstream to the host's real resolver.

That table is populated automatically from your compose file. The **service key**
becomes a resolvable name. This is the complete explanation for:

```yaml
- BACKEND_URL=http://backend:8000
```

`backend` is the service key in `docker-compose.yml`. The engine registered it. The
frontend's resolver asks `127.0.0.11`, gets `172.18.0.2`, connects. There is no
`/etc/hosts` entry, no external DNS, no configuration you wrote.

Two things follow, and both matter later:

1. **This is Docker's feature, not the network's.** Nothing about TCP/IP gives you
   name resolution for free. Docker chose to provide a resolver. **ECS does not
   provide an equivalent by default.** When you move to ECS, `http://backend:8000`
   stops resolving - not because ECS is broken, but because the thing that made it
   work was a local Docker daemon you no longer have.
2. **The old `docker0` default bridge does *not* have embedded DNS**, only
   user-defined bridges do. This is why "use a named network" is standard advice.

---

## 6) Service discovery, defined properly

**Service discovery is the problem of finding a healthy instance of a service when
its address is not knowable in advance.**

The reason it exists: in any dynamic system, addresses are ephemeral. An ECS task
that is replaced gets a **new ENI and a new IP**. Hardcoding is impossible. So
something must map a stable *name* to a current *address*, and keep it updated as
instances come and go.

Every solution has the same three parts:

```
  registration  ->  something records "instance X is at address A, and is healthy"
  resolution    ->  a client asks "where is service S?" and gets addresses
  health        ->  unhealthy instances are removed from the answer
```

The patterns differ in *where the intelligence sits*:

| Pattern | Client does | Cost | Example |
| :-- | :-- | :-- | :-- |
| **None needed** | uses `localhost` | zero | two containers in one task |
| **DNS-based** | resolves a name, connects directly | tiny | AWS Cloud Map |
| **Proxy / LB** | connects to one stable endpoint, which routes | real money | ALB, NLB |
| **Client-side LB** | fetches the full instance list, picks one itself | code complexity | Eureka, Consul |
| **Service mesh** | a sidecar proxy does all of it transparently | high complexity | Istio, App Mesh |

Docker's embedded DNS is the DNS-based pattern, done locally. AWS Cloud Map is the
same pattern, done at cloud scale.

---

## 7) Why DNS-based discovery cannot load-balance your app

This is the most useful thing in this file, and it is the technical core of the
decision in point **2**.

Cloud Map with `MULTIVALUE` routing returns *several* A records for one name, and
resolvers hand them out in varying order. That looks like round-robin load
balancing. It is not, for three reasons:

1. **Resolvers cache for the TTL.** Within the TTL every lookup gets the same
   cached answer.
2. **Clients keep connections open.** Your frontend builds one HTTP client and
   reuses it. It resolves once, opens a TCP connection, and then sends every
   subsequent request down that same connection. It will never look up the name
   again.
3. **DNS has no idea about load.** It cannot know that instance 1 is busy with a
   45-second RAG query while instance 2 is idle. It hands out addresses blind.

So DNS discovery solves **"where is the backend"** (churn), and does *not* solve
**"which backend should get this request"** (distribution). A load balancer solves
both, because it terminates the connection and makes a per-request decision with
live health and load information.

**The consequence for FinSights.** Your backend runs `uvicorn --workers 1` and each
query takes 30-50 seconds. If you ever needed to serve two users at once you would
need multiple backend tasks *and* something to distribute across them - which is a
load balancer, which is ~$16.43/month, which is off the table. So Cloud Map's
independent-scaling benefit is unreachable for you: **the only advantage of the
two-service design requires the component you have ruled out.**

That is why one task with two containers wins. Not because it is a compromise -
because the alternative's benefit is unavailable at your budget.

---

## 8) Load balancers, since you asked in the header

A load balancer is a **reverse proxy**: a server that accepts client connections,
then makes its own connections to backends and relays.

Because it terminates the connection it can do things DNS structurally cannot:
per-request routing, active health checking, TLS termination, path- and
host-based routing, connection draining, sticky sessions, retries, and metrics.

AWS's three:

| | Layer | Routes on | Notes |
| :-- | :-- | :-- | :-- |
| **ALB** | 7 (HTTP) | path, host, header, method | WebSocket capable; ~$16.43/mo + LCU |
| **NLB** | 4 (TCP) | ip/port only | extreme throughput, static IP |
| **CLB** | legacy | - | do not use |

For a Streamlit app you would need an ALB (Streamlit needs WebSockets, and ALB
supports the HTTP Upgrade handshake; also sticky sessions, since a session is
pinned to one process).

**The real value an ALB buys you is a stable DNS name.** Without one, your
frontend lives at the task's public IP, which changes every time the task is
replaced. You cannot put an ephemeral IP on a resume, in a slide, or in a
bookmark. That - not load balancing - is the reason to want one eventually. It
also gives you a hostname you can put a certificate on, i.e. HTTPS.

Worth being precise about the money: the ALB's ~$16.43/month is **more than your
entire frontend container costs**. That asymmetry is normal in AWS and worth
internalising - the fixed-price managed components often dominate the bill of a
small system. This is the same lesson as the NAT gateway in
[[S02c - AWS Substrate - EC2, VPC, Subnets, NAT, ECR]].

---

## 9) Carry-forward

1. A connection is a four-tuple; a port space belongs to a **network namespace**,
   not a machine.
2. `localhost` means "this network namespace" - which is why co-located containers
   need no discovery at all.
3. Docker gives containers a bridge (software L2 switch) plus veth pairs, and an
   **embedded DNS resolver at `127.0.0.11`** that makes compose service names
   resolvable. ECS provides no equivalent.
4. Published ports are iptables DNAT, not a moved listener.
5. Service discovery = registration + resolution + health.
6. **DNS discovery solves churn, not distribution.** Load balancing needs
   something that terminates connections.
7. An ALB's real product is a stable name and TLS, and it costs more than a small
   container.
