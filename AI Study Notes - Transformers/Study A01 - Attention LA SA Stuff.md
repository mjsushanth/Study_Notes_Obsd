
![[Pasted image 20260212105156.png]]


![[Pasted image 20260212105308.png]]


Sparse attention: fixed selection, sliding window, or a special range chosen for interaction. or even, selective set of global tokens used sometimes. 

![[Pasted image 20260212105426.png]]

Linear attention is where every new incoming token updates this shared struct of memory of representation of tokens in compressed space. No more pairwise.


![[Pasted image 20260212105518.png]]

KIMI 2.5 and DEEPSEEK.

![[Pasted image 20260212105558.png]]


![[Pasted image 20260212105627.png]]


Writing a player in the direction first compresses the representation of every single token unit before the comparison event happens. There is no accumulation. There is just individual compression at scale. 

The fundamental scaling problem is past 256k token limit window where below 256k both linear compressed retention and also sparse retention both seem to work fine but sparse retention still has problems with forgetting the tokens and forgetting relevance.


![[Pasted image 20260212110206.png]]


![[Pasted image 20260212110222.png]]


Minimax M1 >> research.
MLA GQA costlier -- scales quadratically but one is quadratic cheaper degree, one is higher. Minimax scales linearly but still has problems.


![[Pasted image 20260212110442.png]]

![[Pasted image 20260212110325.png]]

## **Minimax** M2 - gave up linear lightning methods. 

![[Pasted image 20260212110529.png]]


