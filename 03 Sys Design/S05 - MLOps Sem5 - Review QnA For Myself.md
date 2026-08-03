
Q1: Why semantic variants + window expansion over section-level adaptive windows?
- simplicity with measurable impact over architectural complexity. Section-level adaptive windows sound sophisticated, but they introduce three critical failure modes:
- section boundaries are fuzzy in 10-K filings. Our analysis of 4,674 companies showed that sections often blend together - risk factors bleed into MD&A, business descriptions reference financials. A rigid section-based window would create artificial boundaries
- semantic variants cost $0.00003 per query - three Haiku API calls generating alternative phrasings. The benefit is concrete: we retrieve different result sets from the same semantic space, increasing recall by approximately 15-20% based on our duplicate analysis.
- Section-adaptive windows would require complex heuristics
- window expansion is deterministic and debuggable. When a query fails, I can trace exactly which sentences were retrieved and why. Section-adaptive logic creates a black box where debugging becomes 'why did the heuristic choose this boundary?' 
- principle: prefer simple, measurable strategies over complex, unvalidated ones. I can A/B test variant counts (2 vs 3 vs 5) in an afternoon.


Q2: Why hybrid KPI + Vector instead of unified vector database?
- KPIs and narratives have fundamentally different mathematical properties that don't belong in the same retrieval space.
- (1) exact values matter - $168.0B vs $167.9B is a meaningful difference, (2) they're queryable by discrete dimensions - company × year × metric, and (3) they form a complete, dense table where every company-year should have values.
- Narratives are unstructured semantic content: 'Management discussed supply chain challenges affecting gross margins.' These need (1) embedding similarity to match intent, (2) window expansion to preserve context, and (3) sparse coverage - most companies don't discuss every topic.
-  '$168B revenue' semantically close to '$150B revenue'; loses value in semantic space.
-  hybrid approach gives me business logic in SQL style, semantic understanding in vectors
-  unified Pinecone instance would cost $70-700/month and still require post-filtering for exact KPI matches.

Q3: justify 1M sample vs 71.8M full corpus?
- sampling strategy isn't arbitrary - it's business-driven representativeness based on query patterns we expect
- stratified approach uses three temporal bins: Modern (2018-2025), Middle (2013-2017), Historical (pre-2013). 
- each bin, we balanced by company and section to avoid over-representing verbose filers.
-  95% of realistic financial queries reference the last 3-5 years. Our Modern bin has complete coverage for 21 companies across 2018-2025. The Middle and Historical bins provide trend context but don't need exhaustive coverage.
- Confidence level: High for modern queries, medium for historical deep-dives. 
- embedding all 71.8M sentences - would cost $7,180 in embedding fees (71.8M × $0.0001 per 1K tokens) and $350/month in vector storage. 
- optimize for the common case, acknowledge edge cases explicitly.



Q5: Why S3 Vectors instead of Pinecone/Weaviate in enterprise setting?
- Even without cost constraints, I'd still choose S3 Vectors - but the reasoning changes.
- In the student project: cost was primary. S3 Vectors at $0.18/month vs Pinecone at $70-700/month.
- Enterprise already has S3, IAM policies, backup procedures, security controls. Adding Pinecone means new vendor contracts, new security reviews, new PII compliance checks. S3 Vectors piggybacks on existing infrastructure.
- Parquet files are portable. If S3 Vectors becomes inadequate, I can migrate to Qdrant, Weaviate, or even Pinecone by reading my Parquet files and re-uploading.
- Query volume > 10,000/day (S3 Vectors performance degrades)
- Need < 50ms P99 latency (S3 Vectors is ~200-300ms)
 

Q7: No comprehensive latency tracking - how will you optimize?
- Phase 1: Instrumentation Design (Already mentally planned) `@track_latency('entity_extraction')`
- log to a `latency_metrics.parquet` with schema: P50, P95, P99 latencies per stage
- Budget constraints (can I use faster models?)
- Principle: measure before optimizing. Optimize what matters to users, not what's theoretically slowest.


Q8: How would this handle 100 concurrent users? Scaling strategy?
- Current architecture is single-threaded, synchronous, designed for analyst workflows - not high-concurrency web traffic.
- S3 Vectors Query (200-300ms): S3 Vectors can handle ~100 QPS, but my code doesn't leverage it
- Bedrock API Limits (5-8 seconds per query): 10-20 concurrent requests, throttling errors, need to request quota increase + implement backoff/retry.
- Query Logger: Not thread-safe. At 100 concurrent: Race conditions.
- asyncio.gather() for parallel component calls, Redis for query result caching, logger to DynamoDB (append-friendly).
```
Architecture change needed:
- Deploy to ECS/EKS with auto-scaling (not Lambda - too expensive at high volume)
- Add Redis/ElastiCache for caching
- Use SQS queue for rate limiting
- Separate vector retrieval into dedicated service
- Add CDN for static responses
```
- Design for your actual SLA, not theoretical maximum scale.


Q9: What happens at 10M sentences? When does your architecture break?
- Architecture has three distinct breaking points at different scales, each requiring different solutions:
  - S3 Vectors Retrieval Latency (~5M vectors): lil higher.
  - switch to managed HNSW indexes (Pinecone, Qdrant). Cost increases to $70-200/month, but latency drops to ~100ms.
  - Memory Footprint in Lambda breaks if too much: go elsewhere.
  - Embedding Cost at Full Corpus (~50-100M sentences)

Q10: No systematic evaluation - how do you know it works well?
- have evaluation infrastructure (BERTScore, ROUGE-L, Cosine Similarity) but no systematic results because I haven't run the full evaluation harness. 
- What I Know (Qualitative Evidence): Factoid KPI queries work perfectly, narrative queries work well.
- Complex multi-company comparisons work reasonably well.
- Missing Quantitative Evidence: Retrieval Recall. Variant Effectiveness. 
- ?? 50-100 human-labeled query-answer pairs / Ground truth sentence IDs mapped to each query / Multiple annotators for inter-rater reliability.

Q11: Entity extraction precision/recall - do you have measurements?
- Unlike NER for person/location/organization, financial entity extraction has ground truth. 
- Tested ~50 queries across patterns: High confidence (90-95% accuracy).
  - Query: 'What was Apple revenue in 2020?'
    - Clearly refers to: AAPL, 2020, revenue ✅
  - Query: 'How did tech giants perform in recent years?'
    - 'tech giants' = AAPL, MSFT, GOOGL, AMZN, META? Or also NVDA, TSLA?
    - 'recent years' = last 2 years? 3 years? 5 years?
    - No single correct answer - this is fuzzy matching by design
- 'Manual testing across 50 queries shows it handles explicit entities well, struggles with ambiguous references, and has known edge cases I've documented.'
- System is qualitatively validated, quantitatively unproven.'"



Q12: Haiku vs Sonnet/Opus - answer quality sacrifice for cost?
- Haiku 4.5: $1/M input, $5/M output → ~$0.01/query (9K tokens avg)
- Sonnet 4: $3/M input, $15/M output → ~$0.03/query (3x more expensive)
- Opus 4: $15/M input, $75/M output → ~$0.15/query (15x more expensive)
- **The principle: optimize for cost until users complain about quality.** I'd rather launch cheap and upgrade than over-engineer quality no one notices.

Q13: Why ±3 semantic variants? Justify choice.
- Why I Chose ±3 Without Research:
  - Time pressure + diminishing returns + qualitative confidence. From manual inspection, ±3 felt like the sweet spot. Could ±4 be slightly better? Maybe. Is it worth a week of experiments to find out? Not when the system isn't deployed yet.
  - I treated this as an engineering problem, not a research problem. That's pragmatic, but it means I can't cite empirical evidence for the ±3 choice.
- Quick experiments of window: `window_sizes = [0, 1, 2, 3, 5, 7, 10]`, bertscore, measure_irrelevant_content() - figure out plots of quality vs window size.
- diminishing returns + qualitative confidence. I treated this as an engineering problem, not a research problem. That's pragmatic, but it means I can't cite empirical evidence for the ±3 choice.


Q14: Why pure semantic search? Why no BM25 hybrid retrieval?
- deliberate under-exploration. RAG literature consistently shows hybrid retrieval (semantic + keyword) outperforms pure semantic.
- BM25 catches exact term matches, Semantic embeddings catch synonyms and paraphrases. RRF (Reciprocal Rank Fusion) combines both signals.
- BM25 requires inverted index over 1M sentences: Index is large (~500MB-1GB for 1M sentences), Need to rebuild on corpus changes.
- S3 Vectors Already Has Metadata Filtering: **approximates** what BM25 does - pre-filtering to relevant company/year/section before semantic search. 
- would **term-level BM25** inside those filtered results improve quality enough to justify the complexity?


Q15: Solo developer - what would you do differently?
- Person A: KPI extraction pipeline (uses SQLite, zero cloud dependencies)
- Person B: Embedding pipeline (standalone script)
- Person C: Frontend UI (mocked backend API)
- Learn: isolated function development, module development, and isolated module demo.


Q21: Research that i MISSED while development and saw later:
- I didn't read the latest RAG papers on hybrid retrieval (ColBERT, SPLADE, dense-sparse fusion). If I had, I might have found clever implementations that are easier than classic BM25. But time-boxing literature review to focus on implementation is a valid choice when you're one person.





