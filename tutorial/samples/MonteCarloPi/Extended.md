***

### Theoretical Foundations

Monte Carlo simulation methods are computational algorithms which rely on repeated random sampling to estimate a results. Monte Carlo simulation can be used to calculate the value of $\pi$ as follows

$$
\dfrac{\text{area of unit circle}}{\text{area of } [-1,1]^2} = \dfrac{\pi}{4} = 
    \dfrac{\text{hits in unit circle}}{\text{randomly generated points in } [-1,1]^2}.
$$

More precisely if $A \subset \mathbb{R}^n$, $f : A \rightarrow \mathbb{R}$ is an integrable function and $x^{(i)} \in A$, $i=1, \ldots, n$ are uniformly distributed points then

$$
\int_A f(x) dx \approx \frac{\mathrm{vol}(A)}{n} \sum_{i=1}^n f(x^{(i)}).
$$ {#eq:mc-integral-approx}

Now apply ({@eq:mc-integral-approx}) with $A = [-1,1]^2$ and $f(x) = \mathbb{1}_{\{x_1^2 + x_2^2 \leq 1\}}$ the indicator function of the unit circle. 


### Implementation

Random points in the unit square are generated and we calculate how many of them are inside the unit circle. Given an execution context, which is either a CPU or a GPU device we allocate buffers for the generated points and a scalar to the simulated value of $\pi$. We define aa transformation that checks if point is inside unit square or not. The value 4.0 is because we only simulate points in positive quadrant. The actual compuations happen in the `for` loop where we iterate over multiple batches, generate random numbers, apply the transformation to count the number of points inside the unit circle followed by a mean reduction.

```{.cs}
var points = ctx.Device.Allocate<double2>(Shape.Create((long)batchSize));
var pi = ctx.Device.Allocate<double>(Shape.Scalar);

var pis = Map(points, point => (point.x * point.x + point.y * point.y) < 1.0 ? 4.0 : 0.0);

for (var i = 0; i < batchs; ++i)
{
    Console.WriteLine($"Batch {i}");
    var offset = batchSize * (ulong)i;
    ctx.Assign(points, RandomUniform<double2>(seed, offset));
    ctx.Assign(pi, i == 0 ? ReduceMean(pis) : (pi + ReduceMean(pis)) / 2.0);
}
```


