# DB Indexing & Sharding — Deep Dive

*Expanded from my own notes and questions below. Original questions kept verbatim as blockquotes; answers are the deep dives.*

---

## Part 1 — Indexing

![[Pasted image 20260816203942.png]]

The diagram above is the whole indexing decision in one picture: **efficient access needed? → is the table big enough to matter? → what shape is the query?** Everything below is *why* each branch of that tree ends where it does.

### What an index actually is

An index is a **separate, auxiliary data structure** that maps values in a column to the physical location of the rows containing them, so the engine can jump straight to matching rows instead of reading every row on disk.

That's the entire trade-off, stated plainly:
- **Cost:** extra storage, and extra write work on every `INSERT`/`UPDATE`/`DELETE` (the index has to be kept in sync with the table, forever).
- **Benefit:** reads on the indexed pattern go from scanning everything to jumping directly to what matters.

> Every index is a bet that you'll read that access pattern often enough to justify paying for it on *every single write*. Over-indexing is a real anti-pattern — it quietly taxes all your writes for read patterns nobody actually uses.

A **full table scan** is the baseline: `O(n)`, no shortcut structure, inspect every row. Fine for small tables; ruinous once a table stops fitting in memory and every scan becomes disk-bound.

---

### B-Tree (really: B+Tree)

> **My note:** *btree - standard, good with = or > < range indexing, good with prefix - start based index. sorted. standard in most db.*

What real databases (Postgres, MySQL/InnoDB, Oracle, SQL Server) actually ship as the default index is specifically a **B+Tree**, not a plain B-tree. Two design choices make it the generalist workhorse:

1. **Huge fanout, tuned to disk page size.** Each internal node holds hundreds of keys, not two (unlike a binary tree). That keeps the tree's *height* absurdly small — often 3–4 levels even over a billion rows — which means very few disk page reads per lookup. On real hardware, disk seeks are the actual cost driver, not comparisons, so this is the entire point of the design.
2. **Sorted keys + linked leaf level.** Every node stores keys in order, and critically, B+Trees chain all the *leaf* nodes together in a linked list. Once you find where a range starts, you just walk forward through leaves — no need to climb back up the tree.

That sorted, linked structure is *exactly* why B-trees are the only one of the four index types that natively supports:
- **Range queries** (`WHERE age > 25`)
- **`ORDER BY`** (data is already sorted)
- **Prefix matching** (`WHERE name LIKE 'Jo%'` — all "Jo..." names are adjacent)

Cost is `O(log n)` for lookup/insert/delete — not constant, but with fanout in the hundreds, `log n` is tiny in practice.

```
                    [ 50 | 100 ]                 <- internal nodes:
                   /      |      \                  guide the search,
             [20|35]   [65|85]  [120|150]           don't hold real data
             /  |  \    /  |  \    /   \
           ...        ...        ...             <- leaf level:
                                                     all actual rows live here

Leaf chain (the range-scan trick):
[10,15,20] -> [25,30,35] -> [40,45,50] -> ... -> [150,160,...]
   linked list across leaves = walk forward for any range, no backtracking
```

---

### Hash Index — the "why limited?" answered

> **My note:** *hash index - nice instant lookup but limited. why?*

A hash index runs the key through a hash function and stores it in a bucket at `hash(key) % bucket_count`. Lookup for an exact match is `O(1)` average — hash once, jump straight to the bucket, done. No tree traversal.

Here's the direct answer to "why limited": **the same property that makes it fast is what breaks everything else.** A good hash function deliberately scrambles similar keys into unrelated buckets — that's *how* it spreads load evenly. But that scrambling destroys order, so:

- **No range queries.** `hash(24)` and `hash(25)` can land in completely unrelated buckets. There's no way to ask "give me everything between 24 and 30" without checking every bucket — order information is gone by design.
- **No prefix matching, no `ORDER BY`.** Same root cause: storage position has zero relationship to value order.
- **Collisions need handling** (chaining or open addressing), which degrades toward `O(n)` under skew.
- **Resizing is expensive.** Growing the bucket count means rehashing — a costly, sometimes-blocking operation. *(Worth noticing: this is the exact same "reshuffle nightmare" that sharding hits later in this doc, just at single-machine scale instead of across servers.)*

Good fit: pure equality lookups, especially memory-resident data (Redis, Memcached, in-memory hash joins). Bad fit: anything involving order, range, or "starts with."

---

### Inverted Index — how full-text search gets cheap

> **My note:** *inverted index - full table scans become efficient because words and tokens get word-hashes,*

Slight correction/sharpening of the mechanism: it's not really "word-hashes" (that's the hash-index idea) — it's **inversion**. The natural structure of text is *document → words it contains*. An inverted index flips that, once, at write time, into *word → list of documents containing it* (a **postings list**). That's the "inverted" part.

A search no longer re-reads any document text. It just looks up the token in a precomputed dictionary and gets back the matching document IDs directly.

For multi-word queries ("machine learning"), you look up each word's postings list separately and **intersect** the (sorted) lists — cheap, because both sides are already sorted lists of IDs:

```
Forward (raw documents, how text actually exists):
 Doc1: "machine learning is powerful"
 Doc2: "deep learning models"
 Doc3: "machine code basics"

Inverted (precomputed once, at write time):
 "machine"  -> [Doc1, Doc3]
 "learning" -> [Doc1, Doc2]
 "deep"     -> [Doc2]
 "powerful" -> [Doc1]
 "code"     -> [Doc3]
 "basics"   -> [Doc3]

Query "machine learning":
 [Doc1, Doc3] ∩ [Doc1, Doc2]  =  [Doc1]
```

The engineering underneath this (Lucene/Elasticsearch/Solr territory): **tokenization** (splitting text into terms), **stemming** and **stop-word removal** (deciding what counts as a meaningful term), **term frequency** (for relevance ranking), and **postings-list compression** (delta-encoding sorted ID lists, since they can get huge).

---

### Geospatial Index — the one not in your notes, but in your diagram

B-tree order is fundamentally 1D: sorted means "what comes right before/after." Geographic coordinates are 2D, and "nearby" doesn't correspond to "adjacent in sorted order" — two points 1 meter apart can have wildly different latitude *and* longitude sort positions. So plain B-tree ordering breaks down for "find everything within 5km."

Fix: structures built specifically for spatial locality — **R-trees** (nested bounding boxes, used by PostGIS) or **Geohash/quad-tree** encodings (a space-filling curve that folds 2D proximity into a sortable 1D string prefix, letting you fall back on ordinary prefix-scan tricks — used by Redis `GEO*` commands, MongoDB).

---

### Putting it together

| Index type | Best for | Lookup cost | Key limitation |
|---|---|---|---|
| B+Tree | Range, sort, prefix, general purpose | `O(log n)` | Not `O(1)` for pure equality |
| Hash | Exact-match equality only | `O(1)` avg | No range/order; resize cost |
| Inverted | Full-text / term search | `O(1)` term lookup + intersection | Needs a tokenization pipeline |
| Geospatial (R-tree/Geohash) | Proximity/location queries | ~`O(log n)` | More complex to maintain |

**The one-line framework, matching the decision-tree image exactly:** index choice is dictated by the *shape of your dominant query*, not the shape of your data. Exact key → hash. Range/sort/prefix → B+Tree. Ad-hoc term inside text → inverted. Proximity → geospatial.

One more expert note the diagram doesn't show: a **composite/covering index** (an index across multiple columns matching your `WHERE` + `ORDER BY` combo) can let a query be answered *entirely from the index*, never touching the table at all — an "index-only scan." Worth reaching for once single-column indexes aren't enough.

---
---

## Part 2 — Sharding

### The setup: when one database isn't enough

> **My note:** *Assume 70TB, tons of requests, 1 DB only, all requests going there. Now, increasing from 10k to 40k 60k writes per second. identify, realize scaling and cieling. first attempt - ton ton ton of throughput with just vertical scaling, lot of compute, power, more ram more DB more speed more OCUs, etc.*

Vertical scaling — bigger box, more RAM/CPU/IOPS — is always the first lever, because it needs zero architecture change. It hits a hard ceiling for structural reasons, not just cost:

1. **Physical ceiling.** Cloud providers cap instance sizes. There is a biggest box you can rent, full stop.
2. **Single point of failure never goes away.** One node, however big, is still one node.
3. **Single write path.** Most relational architectures have exactly one primary accepting writes — read replicas don't help write throughput, and bigger CPU/RAM doesn't remove this serialization point.
4. **Superlinear cost.** Doubling specs at the high end often costs far more than 2x — premium hardware tiers aren't priced linearly.

Past a certain scale (your numbers — 70TB, 40–60k writes/sec — are realistically past comfortable single-node territory), the only lever left is **horizontal scaling**: more machines, splitting both data and load across them. That's sharding.

```
VERTICAL SCALING (bigger box)         HORIZONTAL SCALING (more boxes)

   [ Small DB ]                        [Shard1][Shard2][Shard3][Shard4]
        |                                  |       |       |       |
        v                                  +-------+---+---+-------+
   [ Bigger DB ]                                       |
        |                             Each shard handles a SLICE of
        v                             data + load, in parallel. Add
 [ Biggest DB $$$$ ]  <- hard ceiling  another box to add more capacity —
   one machine,                       no single ceiling.
   one point of failure
```

> **My note:** *then comes sharding: process of splitting existing data across multiple independent DBs and making them shards. shards >> ++ increase data capacity and brings operational capacity. shard management. access. knowledge of where data is.*

Formalized: sharding is **horizontal partitioning** — each shard holds a disjoint subset of rows, and something (app layer or routing tier) must always know *which* shard holds a given piece of data. That "knowledge of where data is" is the whole operational trade — you've swapped a hard scaling ceiling for a new distributed-systems routing problem.

---

### Q1 — What to shard by?

The single most consequential, hardest-to-reverse decision in a sharded system. A good shard key needs:

- **High cardinality** — enough distinct values to actually spread across many shards (a boolean caps you at 2 shards, forever).
- **Matches the dominant query pattern** — pick the column most `WHERE` clauses already filter on, so most queries can be answered by one shard (see Q5).
- **Even load, not just even count** (see Q4).
- **Stability** — the value shouldn't change after row creation. A mutating shard key means the row may need to physically *move* shards later, which is expensive and error-prone.

Typical real choices: `user_id`, `tenant_id`, geographic region, `order_id`.

### Q2 — How do we distribute values across a shard?

Three broad strategies:

1. **Range-based** — contiguous key ranges map to a shard (e.g. `user_id` 1–1M → shard 1). Simple, supports in-shard range scans. Danger: monotonically increasing keys (auto-increment IDs, timestamps) mean *all new writes* land on the newest/last shard — a permanent hotspot.
2. **Hash-based** — `hash(key) % N` (or onto a hash ring) picks the shard. Spreads writes evenly regardless of key pattern, but destroys range-query locality — a range scan now has to hit every shard.
3. **Directory/lookup-based** — an explicit key→shard mapping service. Most flexible (can rebalance individual keys at will), but the directory itself becomes a new must-scale, must-be-highly-available component. You haven't removed the hard problem, you've relocated it.

### Q3 — Exploratory queries for cardinality and spread

Cardinality = number of distinct values a candidate key would take. **High cardinality alone isn't sufficient** — a key can have millions of distinct values and still be dominated by a handful of them (a viral product responsible for 40% of all `product_id` traffic, for instance).

Before committing to a shard key, actually check:
- `SELECT candidate_key, COUNT(*) FROM table GROUP BY candidate_key ORDER BY COUNT(*) DESC LIMIT 20` — find the heaviest hitters.
- What % of total rows/traffic the **top 1%** of key values account for — a concentration/skew check.
- Cross-reference against **query logs**, not just stored data volume — a key can have modest data but massive read/write QPS (a small "config" row hit by every request, for example).

Low cardinality (e.g. "country," ~5 values) caps you at a handful of usable shards forever, *and* guarantees imbalance, since real-world categorical data is essentially never uniform.

### Q4 — Should we care about perfectly even distribution, always?

No — and this is the subtle one. The actual goal is even **load**, not even **data volume**. Two different targets:

- **Storage-bound sharding:** goal is even bytes, so no single disk fills up first.
- **Compute/query-bound sharding:** goal is even query traffic, so no single shard's CPU/IOPS saturates. A shard with 5% of the data but 50% of the queries is a *bad* shard even though "data" looks balanced.

Some imbalance is legitimate **by design** — a dedicated large shard for one huge enterprise tenant in multi-tenant SaaS, deliberately isolated so its load can't spill onto everyone else's shared shard. Imbalance by design (isolation) is a different animal from imbalance by accident (bad key choice).

### Q5 — Queries hitting many shards vs. one shard: the tradeoff

- **Scoped (single-shard) query** — the shard key is already in the `WHERE` clause, so the app knows exactly which one shard to hit. Fast, no coordination. This is the entire point of a good shard key: make *most* real queries land here.
- **Scatter-gather (cross-shard) query** — no shard key in the filter (e.g. "top 10 products globally" when sharded by user), so the query fans out to *every* shard, then results get merged/re-sorted at a coordinator. Costs:
  - Tail latency = the **slowest** shard's response, not the average — compounds badly as shard count grows.
  - Merge overhead: paginating/sorting across N result sets is genuinely nontrivial (k-way merge).
  - Every extra shard is one more point of partial failure for that single logical query.

Design implication: accept that a *few* query types will always be cross-shard (admin dashboards, global reports) — route those to a read-replica or analytics warehouse, and keep them away from live transactional shards.

### Q6 — Future-proofing your shard strategy

The single best technique: shard into far more **logical shards** than you have **physical databases** today — say 4096 logical shards, running on just 8 physical instances (512 logical shards per box). Growing means *moving already-existing logical shards* from an old box to a new one — you never re-hash or re-bucket a single row. The row→logical-shard mapping never changes; only the logical→physical assignment does. This decouples "how a row is bucketed" (a decision you basically never want to revisit) from "how many machines I run today" (a decision you want total freedom to revisit).

Second technique — direct segue into Q8–Q10 — is exactly why **consistent hashing** exists: avoid naive `hash(key) % N` from day one, so `N` can change without a full data reshuffle.

Third: build **online/live resharding tooling** (dual-write to old + new location, background backfill, verify, cut over) as first-class infrastructure *before* you need it in an emergency.

Fourth: keep the shard key immutable, per Q1.

### Q7 — Identifying shard-usage/traffic patterns, across domains

Four worked examples:

1. **E-commerce** (shard by `user_id` or `order_id`): a single seller's flash sale means that seller's shard suddenly takes 100x normal traffic. Correction: detect via per-shard QPS monitoring; give that seller/SKU a dedicated shard, or front it with a cache so the shard itself isn't hammered.
2. **Social media** (shard by `user_id`): the "celebrity problem" — a 50M-follower account means one post triggers fan-out disproportionate to a normal user, all funneled at whichever shard owns that one `user_id`. Correction: special-case high-follower accounts with a *pull*-based timeline model instead of push/fan-out-on-write, so their shard isn't hammered on every post.
3. **Multi-tenant SaaS** (shard by `tenant_id`): one enterprise customer can dwarf hundreds of small customers combined. Correction: tiered sharding — dedicated shards for large tenants, a shared pool for small ones, re-evaluated as usage grows.
4. **IoT/time-series** (shard by `device_id` or time bucket): naive time-bucket sharding means *all current writes* land on whichever shard represents "now" — a permanent hotspot on the newest shard. Correction: a **compound key** (`device_id` + time bucket, or hashing `device_id`) so writes spread across shards even within the same narrow time window.

General detection method across all four: per-shard dashboards for QPS, CPU, IOPS, and p99 latency, with an alert when one shard's load exceeds some multiple (e.g. 3x) of the *average* shard's load. That ratio — max-shard-load ÷ avg-shard-load — is a simple, effective running "hotspot score."

---

![[Pasted image 20260816210438.png]]

### Q8 — How did hash-based sharding start, how does it work, what did it fix?

**The problem it fixed:** naive range-based sharding (Q2) hotspots badly on monotonically increasing keys — auto-increment IDs, timestamps. Every new write piles onto the single "current" shard, which defeats the entire purpose of sharding (you added machines, but only one of them does any work). This exact failure mode is what pushed early large-scale sharded deployments toward hashing the key first — a pattern that shows up repeatedly across 2000s–2010s web-scale engineering write-ups (Flickr- and LiveJournal-era MySQL sharding is the canonical example).

**How it works** (matches the left half of the image): run the key through a hash function *before* deciding placement. A good hash function maps sequential/similar inputs to statistically unrelated, uniformly distributed outputs — so two keys created back-to-back (IDs 1000 and 1001) land in completely different, effectively random shards. Writes spread evenly across all shards immediately, independent of the key's real-world pattern.

Mechanism: `shard_index = hash(key) % N`. With `N=3` (as in the image), `hash(key)` is computed once, then reduced mod 3 to pick shard 0, 1, or 2.

**Cost this introduces** (already flagged in Q5): range-query locality is gone. "Find all orders between date X and Y" can no longer be answered by one shard — it now requires scatter-gather across all of them.

### Q9 — Adding new shards: the reshuffle nightmare

Direct confirmation of the intuition: yes, and it's *almost total* data movement, not partial. With naive `hash(key) % N`, changing `N` from 3 to 4 changes the modulo result for the vast majority of keys, because `hash(key) % 3` and `hash(key) % 4` are essentially unrelated computations — there's no structural reason a key's "bucket out of 3" has anything to do with its "bucket out of 4."

```
BEFORE (N=3): hash(key) % 3          AFTER (N=4): hash(key) % 4

  key A -> bucket 1                    key A -> bucket 1   (lucky, stays)
  key B -> bucket 2                    key B -> bucket 3   (MOVES)
  key C -> bucket 0                    key C -> bucket 2   (MOVES)
  key D -> bucket 1                    key D -> bucket 0   (MOVES)
  ...                                  ...

Only keys where (hash(key) % 3 == hash(key) % 4) keep their place —
a small minority. Everything else must be physically copied.
For 70TB, that's a massive, slow, risky live migration.
```

That migration *is* the reshuffle nightmare — and it's exactly what consistent hashing is built to avoid.

### Q10 — What is consistent hashing?

Consistent hashing solves Q9 directly: add or remove servers while moving only a **small, predictable fraction** of the data, instead of nearly all of it.

Mechanism, matched to the ring in the image:

- Both the **data keys** and the **servers** (DB1–DB4) are hashed onto the *same* fixed circular hash space (0–99 in the image; real systems typically use 0 to 2³²−1).
- To find which server owns a key: hash the key to a ring position, then walk **clockwise** until hitting the first server. In the image: `hash(1234) = 16` → walk clockwise from position 16 → first server hit is **DB2** → DB2 owns event 1234.
- **Why this fixes the reshuffle nightmare:** adding a new server only takes over the keys in the arc between itself and its counter-clockwise neighbor — every other server's keys are completely undisturbed, because "walk clockwise until you hit a server" never changes for them. Adding the Nth server to a ring of N−1 moves roughly `1/N` of total keys on average — not nearly all of them, unlike plain `mod N`.
- Removing a server works symmetrically: only its keys move, to its clockwise neighbor.
- **Real-world refinement:** with only a handful of servers, a plain ring can still be lumpy (arcs of very different sizes by chance). Production systems (Cassandra, DynamoDB, Riak) fix this with **virtual nodes (vnodes)** — each physical server gets many positions on the ring (e.g. 256), so the law of large numbers evens out arc sizes even with few physical machines.
- This is architecturally the *same idea* as Q6's "over-shard into logical shards" trick — both decouple "how a key is placed" from "how many machines exist right now."

---

### Sharding strategies, side by side

| Strategy | Mechanism | Pros | Cons |
|---|---|---|---|
| Range-based | Contiguous key ranges → shard | Simple; supports in-shard range scans | Hotspots on monotonic keys |
| Hash-based (`mod N`) | `hash(key) % N` | Even write distribution | Full reshuffle when `N` changes; no range queries |
| Consistent hashing | Keys + servers on one ring | Only ~`1/N` keys move on add/remove | More implementation complexity; needs vnodes for evenness |
| Directory/lookup | Explicit key→shard table | Maximum flexibility, per-key rebalancing | The directory itself must scale + stay highly available |

---

### The throughline

Both halves of this document are the same bet, at different scales. An **index** is a bet on which query pattern you'll run most on one machine. A **shard key** is a bet on which access pattern needs to stay fast as you grow across many machines. **Consistent hashing** is what makes that second bet survivable — it's the reason growing your infrastructure later doesn't force you to undo the first bet you already made.
