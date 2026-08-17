

MC markov property;

We already know the markov property - memory less nature, current state only deps on previous state and not exactly all possible states of the past. however, joint probability still exists in essence in the total chain. 

![[Pasted image 20260816131815.png]]


Priors - are always our beliefs, assumptions about the parameters before we look at the data.
Likelihood - its the likelihood. how well a set of params explains the observed data.
Posterior - which parameter values are likely, given that you observe certain data x.
the hard part is evidence, P(x) ; realistic scenarios, its hard to compute and this is why MCMS enters.


MCMS - chains, stationary distribution, sampling. need to understand more, intuition. 


Initialize , propose , Filter / Acceptance Ratios , Random walk through parameter space , 
Estimations with the samples we have.

![[Pasted image 20260816133927.png]]



![[Screenshot 2026-08-16 at 1.41.12 PM.png]]


Chain gives you samples, Samples give you everything else. 

Uphill and downhill concepts.

MCMC / MCMS.