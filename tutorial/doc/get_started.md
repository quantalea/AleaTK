## Get Started

### Installation
***

#### Prerequisites

Alea TK relies on the Alea GPU compiler and requires a CUDA-capable GPU with **compute capability 2.0** or higher. Alea GPU is free for 
GeForce and Quadro GPUs. Running Alea GPU on Tesla GPUs requires a [license](http://www.quantalea.com/licensing/). 

#### Install from NuGet

Alea TK is best installed as a NuGet package.


### Overview 
***

Alea TK offers two main computing paradigms:

 - **Imperative** computing with tensors and tensor expressions

 - **Symbolic** computing with operators supporting autodifferentiation and back propagation


#### Imperative Computing for General Numerical Applications

The main abstraction of the imperative paradigm is a [tensor](https://en.wikipedia.org/wiki/Tensor) which is a mathematical object to represent linear relations between vectors. Once a basis is selected a tensor can be represented as a multi-dimensional array. 

In Alea TK a **tensor** has a layout defining the number of dimensions, the shape and the strides along each dimension. Shapes may be defined only partially. This is useful if certain input dimensions may change or are not yet known at design time of the computation. The actual values representing the tensor are hold in a buffer. 

The imperative paradigm allows to define calculations in terms of tensor expressions. Tensor expressions are assignable to [l-values](https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue)
and can be used as [r-values](https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue) e.g. for kernel fusion.

Here is an example that calculates $\pi$ with Monte Carlo simulation. Random points in the unit square are generated and we calculate how many of them are inside the unit circle. Given an execution context, which is either a CPU or a GPU device we allocate buffers for the generated points and a scalar to the simulated value of $\pi$. We define aa transformation that checks if point is inside unit square or not. The value 4.0 is because we only simulate points in positive quadrant. The actual compuations happen in the `for` loop where we iterate over multiple batches, generate random numbers, apply the transformation to count the number of points inside the unit circle followed by a mean reduction.


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

The full code for [Monte Carlo Pi](samples/montecarlopi.html) is in the sample gallery.


#### Symbolic Computing for Machine Learning  

The primary objects of symbolic calculations are **variables** and **operators**. A variable assigns an identifier to a future calculation. Alea TK has three variable types: common, parameter, auxiliary (primarily used for temporary results). A variable usually holds two tensors, one to keep the actual values of the variable and a second tensor that holds the gradients in a backward propagation process. An **operator** defines a future computation, which, given input variables, generates output variables. A so called **executor** binds a variable and its computation graph to a computation context, which can be a CPU or GPU device. With an executor it is possible to run forward and backward gradient calculations and to allocate and manage the memory. 

With symbolic computing we can implement various optimization algorithms to train machine learning models represented in terms of a computational graph defined with operators and variables. 

Here is the specification of a convolutional neural network for image classification:

```{.cs}
var images = Variable<float>(PartialShape.Create(-1, 1, 28, 28));

var conv1 = new Convolution2D<float>(images, 5, 5, 20);
var act1 = new ActivationTanh<float>(conv1.Output);
var pool1 = new Pooling2D<float>(act1.Output, PoolingMode.MAX, 2, 2, 2, 2);

var conv2 = new Convolution2D<float>(pool1.Output, 5, 5, 50);
var act2 = new ActivationTanh<float>(conv2.Output);
var pool2 = new Pooling2D<float>(act2.Output, PoolingMode.MAX, 2, 2, 2, 2);

var fc1 = new FullyConnected<float>(pool2.Output, 500);
var act3 = new ActivationTanh<float>(fc1.Output);
var fc2 = new FullyConnected<float>(act3.Output, 10);

var labels = Variable<float>();

var loss = new SoftmaxCrossEntropy<float>(fc2.Output, labels);

```

It uses variables and operators to express the layers of the network. The [MNIST](samples/mnist.html) sample shows how to use such a convolutional neural network for image classification. 


### Highlights
***

It is worth mentioning that Alea TK has some unique features such as 

- Integration of high performance CUDA libraries such as cuDNN, cuBlas and cuRand
- Tensor views and GPU based shuffling for mini batch epochs 
- GPU kernel fusing to reduce the number of kernel launching 


### Next Steps
***

- Read the [tutorials](tutorials.html) to get a deeper understanding of Alea TK and how it can be used to develop and train machine learning models

- If you are looking for self contained examples as starting point for developing new models check out the [sample gallery](gallery.html)

- The [how to](how_to.html) section addresses topics such as how to extend Alea TK or how to contribute to the project

