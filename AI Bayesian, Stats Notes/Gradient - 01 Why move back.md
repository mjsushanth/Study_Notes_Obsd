

At the most basic level, the loss `L(θ)` is a scalar field over parameter space: for each parameter vector `θ`, you get one real number measuring “how bad” your model is. 

- The **gradient** `∇L(θ)` is the vector that tells you, at that point, which direction in parameter space increases the loss the fastest. This is a precise statement: if you look at tiny moves `Δθ`, the first-order Taylor expansion says  `L(θ + Δθ) ≈ L(θ) + ∇L(θ) · Δθ`.  
- The dot product `∇L(θ) · Δθ` is largest (for a fixed step size ‖Δθ‖) when `Δθ` points in the same direction as the gradient. So the gradient vector is literally “direction of steepest ascent” of the loss.

Once you see that, the negative-gradient rule becomes almost tautological: if `+∇L` is “go uphill fastest,” then `−∇L` is “go downhill fastest.” Gradient descent chooses updates of the form `Δθ = −η ∇L(θ)` with a small step size `η > 0`. Plugging this into the expansion gives  
`L(θ + Δθ) ≈ L(θ) − η ‖∇L(θ)‖²`,  

which is guaranteed to decrease the loss for small enough `η` as long as the gradient is nonzero, because `‖∇L(θ)‖²` is always positive. Backpropagation is just the mechanism that efficiently computes `∇L(θ)` for deep networks via the chain rule; once you have that gradient, the “move in the negative direction” rule is exactly the local best move to reduce loss as quickly as possible in parameter space.