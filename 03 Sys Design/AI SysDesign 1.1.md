
Toolcall - Repo exploration, Specific file relevance, task at hand.
MCP - Vendor specific features to context.
RAG - external data. 

![[image-6.png]]

- Context engineering early era - Context summarization as the task tops up, and its usually one-shot. Effectively, agent bound by summarization of its own work, oversimplify/assume based on context window management. ( or rather mismanagement. )
- Answer: fix using sub agents, hierarchical agents. 


## Agent harnessing.

![[image-7.png]]

- Critical: harness engineering - loops. waterfall. each iteration - fresh, clean set of context under how agent should start and finish. strict rule.
- Simple architecture.
- Doesnt deprecate context/prompt engineering. 'Reminds' the coding agent who they are.
- Paradigm on env - Large requirement file, looping on singular task - finish process. Each iteration: fresh laws, fresh rules, fresh context. Feature after feature until completion. 

----

### Inference is hard:


1. CPU / OS is given the task to talk to RAM - SSD, without overloading the RAM or doing any memory bifurcation or copy, but rather talk using MMAP - logical addr. 
2. Thats how model weights are kept huge GB in SSD; and OS ensures RAM - SSD communication. Lazy loading. 
3. RAM competitive space: so weights get eviced, weights get reloaded, evicted.. you see the problem.
4. Assume GPU is doing heavy lifting: matmul, calculations .. then you need one more round of copying from RAM to GPU as well.
5. 