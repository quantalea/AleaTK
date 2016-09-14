## Design Details

### Motivation
***

We explain the Alea TK design decisions by considering an specific tensor calculation: to execute a tensor expression such as the sigmoid `1 / (1+exp(x))` three steps are required:

  1. Element-wise exp on tensor `x`
  2. Broadcast scalar `1` to the shape of `x`, then element-wise `add`
  3. Broadcast scalar `1` to the shape of `x`, then element-wise `div`

Frameworks such as Tensorflow, Theano, mxNet etc. use a hybrid language solution. The operators exp, add, div etc. are implemented in C++ or CUDA C++. Tensor expressions are combined in Python, which invokes the C implementations. 

Disadvantages:  

  - Many small kernel launces since each operator is created statically in C/C++.
  - To pass the intermediate result (e.g. result of `exp(x)`), kernels store results in temporary memory and following kernels will read it back. Since the kernels usually are simple basic operators to obtain flexibility, the computation is small and the memory load/store becomes the main performance bottleneck.
  - One solution these frameworks pursue is to create fused kernels with C++ templates and meta template programming. Extensions require a recompilation of the whole framework
  This increases development complexity and reduces agility for rapid prototyping a lot.


<img src="images/mxnet.png" alt="Sigmoid from MXNet">

Example: Sigmoid from MXNet using C++ templates


The JIT compilation of Alea GPU provides an alternative solution:

  - Compose operations with nested delegates
  - At first execution fuse delegates into a single kernel and JIT compile kernel
  - Use caching to optimize repetitive JIT compilation 


### Value – Expression – Tensor
***

A `Value` represent data. To support kernel fusion, we further classify a `Value` into two subclasses: 

  - `LValue` – represents left value: access to data memory through pointer, can read/write elements through pointer
  - `RValue` – represents right value: no underlying memory, can only read elements, implemented with a delegate of `Func<long, T>`, which returns the element for an index

An `Expression` represents an algorithm or a computation. During assignment, expressions take the input values and generate the output value. We have `LExpr` and `RExpr`:

  - `LExpr` – expression generating `LValue` during assignment
  - `RExpr` – expression generating `RValue` during assignment

A `Tensor` has memory to hold its data and implements both `LValue` and `RValue`. A `Tensor` is also an `RExpr`, which can generate a data reader delegate for further kernel fusion. A `TensorReader` is a helper class that provides a data read delegate and only implements an `RValue`. The basic form of the imperative programming model for tensor computation is: 

```{.cs}
context.Assign(LValue, Expr)
```

During assignment the AleaTK framework allocates a `Tensor` as `LValue` for each `LExpr`, which is calculated by its `Execute` method. The `RExpr` returns a `TensorReader` for fusing with the next expression. The procedure is optimized to allocate minimal memory. For example

```{.cs}
context.Assign(tensorA, tensorB)
```

is just a copy. The output is already set by `tensorA`, which is an `LValue` and `tensorA` is set for the output of `tensorB`. Allocaton for `tensorB` can be bypassed.

<img src="images/class_hierarchy.png" alt="Class Hierarchy">

Image: Tensor, Value and Expression class hierarchy of Alea TK.


### Assignment
***

The class `Expr` is an abstract class. The derived class has to implement `Execute` and `Prepare`:

<img src="images/expr_interface.png" alt="Expr Interface">

An assignment of an expression to an `LValue` requires two passes through the underlying expression tree:

  1.  For every `Expr` the `Prepare` method is called to set the requirement flags for that expression. For example, the `Dot` expression requires that both operands are of type `LValue` because it uses cuBLAS to do the calculation and only an `LValue` can get its memory. Usually, an `LExpr` requires the that the result is an `LValue` which is allocated by the framework. Other requirements can also be imposed such as the layout of the input operands.
  2.  For every `Expr` the `Execute` method is called to do the actual computation. The `LExpr` launches a kernel which writes the result to the output memory. The `RExpr` creates a nested delegate and passes it to the next expression. The actual kernel creation is delayed.

For more details, check the implementation of `RExpr` and `LExpr`. The `RExpr.Execute` method gets the `RValue` for its derived class first (the data element read delegate). If required, it allocates memory and performs the assignment. The `LExpr.Prepare` require the output of this expr to be `LValue`, and do allocation and assignment. The broadcasting also happens during assignment: the delegate behind an expression extends the elements as required by the target shape.

### Map Expr
***

To implement a sigmoid function operating on tensors we can use the `Map1Expr`:

```{.cs}
public class Map1Expr<TInput, TResult> : RExpr<TResult>
{
    public Map1Expr(Expr<TInput> input, Func<TInput, TResult> transform, string opCode = OpCodes.Map1)
    {
        Shape = input.Shape;
        OpCode = opCode;
        Input = input;
        Transform = transform;
        AddOperand(Input);
    }
 
    public Expr<TInput> Input { get; }
 
    public Func<TInput, TResult> Transform { get; }
 
    public override Shape Shape { get; }
 
    protected override IRValue<TResult> GenerateRValue(Assignment assignment)
    {
        var device = assignment.Context.Device;
        var input = assignment.GetInput(Input).ToRValue();
        var transform = Transform;
        var layout = input.Layout;
        var inputRawReader = input.BufferReader.RawReader;
        Func<long, TResult> rawReader = i => transform(inputRawReader(i));
        return new TensorReader<TResult>(device, layout, rawReader);
    }
}
```

This expression accepts as input a single `RValue` and a transform delegate, and returns a new `TensorReader`. Note there is no kernel launched. It just creates a new delegate and the effective kernel creation is delayed. To simplify usability, there is a helper method: 

```{.cs}
public static Expr<TResult> Map<TInput, TResult>(Expr<TInput> input, Func<TInput, TResult> transform, string opCode = OpCodes.Map1)
{
    return new Map1Expr<TInput, TResult>(input, transform, opCode);
}
```

We can now code up the sigmoid function for tensors using the `Map` function:


```{.cs}
var input = … // allocate input tensor then set its values
var output = context.Device.Allocate<float>(input.Shape);
context.Assign(output, Map(input, x => 1.0f / (1.0f + Exp(x)));
output.Print();
```

The `Map` helper generates one expression instance. During assignment the output is set to output tensor and only one kernel is launched. The Alea TK framework provides many basic operators such as `exp`, `add`, `div` etc. The `ExprRegistry` is used to provide generic implementations. The expressions for different types are registered in the static constructor of the `Library` class. Here is an example which registers the `add` operation:

```{.cs}
ExprRegistry.Register<double>(
    (exprParam, inputExprs) => 
        Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), ScalarOps.Add, OpCodes.Add),
        OpCodes.Add, typeof(double), typeof(double));
 
ExprRegistry.Register<float>(
    (exprParam, inputExprs) => 
        Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), ScalarOps.Add, OpCodes.Add),
        OpCodes.Add, typeof(float), typeof(float));
```

The `Exp` method can be created generically by asking the registry to create the expression:


```{.cs}
public static Expr<T> Exp<T>(Expr<T> a)
{
    return ExprRegistry.Create<T>(OpCodes.Exp, a);
}
```

We can now write the expression for the sigmoid function more compact:


```{.cs}
var input = … // allocate input tensor then set its values
var output = context.Device.Allocate<float>(input.Shape);
var one = 1.0.AsScalar<float>();
context.Assign(output, one / (one + Exp(input)));
output.Print();
```

Note that, although the expression is constructed from three smaller expressions, the actual kernel is only created for the outermost expression. The inner expressions are `Map1` or `Map2` expression of type `RExpr`, generating only a new data read delegate. Kernel fusion can take place and at the end only one kernel is invoked.

### Other Expressions
***

The `Reduce` expression is an `LExpr`. It is implemented directly in C# in such a way that it can take the input from a delegate. Hence the output of a `Reduce` expression is allocated but the input does not need to be an `LValue`. Expressions such as


```{.cs}
context.Assign(scalarTensor, ReduceSum(a + b));
```

will also fuse to a single kernel: `a + b` returns an `RExpr` and `ReduceSum` fuses the delegate of `a + b` with the reduce kernel. The `Dot` expression uses the cuBLAS library, which implies that the input and the output have to an `LValue` in order to access the memory through a pointer. The following requirement in requested in the `Prepare` method:


```{.cs}
public override void Prepare(Assignment assignment)
{
    assignment.RequireOutputLValue(A);
    assignment.RequireOutputLValue(B);
    assignment.RequireLayoutFullyUnitStride(A);
    assignment.RequireLayoutFullyUnitStride(B);
    base.Prepare(assignment);
}
```

The input is requested to be an LValue with a memory layout of unit stride. As a consequence, if the input is for example a partial column slice, a copy to a contiguous memory area is necessary so that the cuBlas `Dot` expression can be used. The following expression 

```{.cs}
context.Assign(output, Dot(A+B, C));
```

requires two kernels. First `A + B` is a `RValue`, but the `Dot` expression requires an `LValue`. The framework allocates temporary memory for `A + B` and assigns the `RValue` of `A + B` to it. The second kernel is the cuBLAS matrix multiplication.

### Symbolic Layer
***

The expression and tensor system follows an imperative programming model. This provides more flexibility. For example, it allows in-place modifications:


```{.cs}
var tensor = …
for (var i = 0; i < n; ++i)
{
    context.Assign(tensor, tensor + 1.0.AsScalar<float>());
}
```

The expression system does not support automatic differentiation. This is the purpose of the symbolic layer. Each `Differentiable` operator implements a `Forward` and `Backward` method. A `Tensor` is accessed through a `Variable` and the forward and backward propagation result is calculated using the expression system. This also allows to define bigger symbolic operators which rely on specific mathematical simplifications or performance optimizations. One example is an operator that combines the softmax and cross entropy calculation:

```{.cs}
public override void Forward(Executor executor)
{
    var z = executor.GetTensor(Input);
    var y = executor.GetTensor(Label);
 
    executor.AssignTensor(M, ReduceMax(z.Reshape(-1, z.Shape[z.Shape.Rank - 1]), true, 1));
    var m = executor.GetTensor(M);
 
    executor.AssignTensor(N, z - m - Log(ReduceSum(Exp(z - m), true, 1)));
    var n = executor.GetTensor(N);
 
    executor.AssignTensor(Loss, -ReduceMean(ReduceSum(y*n, 1)));
 
    executor.AssignTensor(Pred, Exp(n));
}
 
public override void Backward(Executor executor)
{
    var p = executor.GetTensor(Pred);
    var y = executor.GetTensor(Label);
    executor.AssignGradient(Input, p - y);
}
```

In the backward computation the gradient is calculated by evaluating the simplified expression `p – y`, instead of going through the full expression tree. 
