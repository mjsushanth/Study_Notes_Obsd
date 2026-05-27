

![[image-11.png]]

![[image-10.png]]


1. PreNorm Dilution problem. ( Same problem in RNN compression - single state problem. )
	1. Earlier layer details are often progressively lost + Newer layers dominate, Repeated distilled version. Nuances from earlier version lost. 
	2. Called - Signal dilution upon depth. ( Accumulated signal across layers. )
	3. Another reason, very large outputs and representations. 


![[image-12.png]]


1. Selective Retrieval across depth of network, Rotating attention by 90.
2. Apply attention vertically across layers, 'Not only attn across tokens, but attn across layers'. 
3. i.e. , take input - RESIDUALS or SKIP connections to carry connections backward or forward. but mostly not backward in transformers. 


#### "Layers" form a sort of query, thereby having their own scores. 

1. So now, attention how it selectively can focus 'tokens', similarly we can workout a mechanism where layers have their selective importance. 
2. "Input dependant" routing and mechanism.
3. N layers - N layer vectors, still have scaling problem to keep in  mind.
4. **Kimi proposed *block attention residuals* - layers became like blocks, groups.** 


![[image-13.png]]

![[image-14.png]]

![[image-15.png]]