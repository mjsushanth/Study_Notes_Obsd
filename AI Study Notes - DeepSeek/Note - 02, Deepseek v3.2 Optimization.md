
DSA - DSA refers to DeepSeek Spark Attention. The aim is repeated from the abstract that we want to reduce the true cost of full attention and we need to achieve high computational efficiency, accuracy, proper reasoning and performance while also not suffering this cost of long contact scenarios. This is where they introduce DeepSeek Spark Attention. One, Scalable Reinforcement Learning Framework. Two, which is another core model. And finally, other items such as large-scale agentic task synthesis pipelines. This refers to the fact that it has to do with integrating reasonings into tool use scenarios and developing a synthesis pipeline around that.

![[Pasted image 20251205133516.png]]


- Maybe you don't need to really look at every single token of the queries in the preceding parts or consecutive parts in the sentence context when you provide large inputs or large PDFs. 
- Preemptively calculate the relevance or relevance factors and scores using a lower precision models and then maintain an index.
- This index of every pair strategy will hold and serve only the top 2048 relevant tokens.
- This way attention is cheaper. Sparse attention. Cost of inference - low.



### 2 - Training Method Advancements:  Dense Warm-Ups, Sparse Training Stage, Top Key Selectors.

![[Pasted image 20251205133737.png]]



### Other advancement: Thinking in tooluse.



![[Pasted image 20251205133815.png]]


