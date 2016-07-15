## Tutorials

### MNIST 
***

#### Multinomial Logistic Regression 

The [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) model is the most simple classifier for categorical dependent variables. Using the Alea TK symbolic layer the model is represented as follows:

```{.cs}
public static Model MultinomialRegressionModel()
{
    var images = Variable<float>();
    var labels = Variable<float>();
    var w = Parameter(Fill(Shape.Create(28 * 28, 10), 0.0f));
    var b = Parameter(Fill(Shape.Create(10), 1.0f));
    var y = Dot(images, w) + b;
    return new Model() { Loss = new SoftmaxCrossEntropy<float>(y, labels), Images = images, Labels = labels };
}
```

The differentiable operator `SoftmaxCrossEntropy` applies softmax and cross entropy loss at once so that it can better optimize the calculations. To actually use the model we build an execution context and an optimizer. 

```{.cs}
var model = MultinomialRegressionModel();
var ctx = Context.GpuContext(0);
var opt = new GradientDescentOptimizer(ctx, model.Loss.Loss, eta);
opt.Initalize();
```

To access the training data we use a `Batcher` instance.  

```{.cs}
var mnist = new MNIST();
var batcher = new Batcher(ctx, mnist.TrainImages, mnist.TrainLabels);
```

We train the model with a basic [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) optimizer using a fixed learning rate `eta` and multiple epochs. For each epoch the data is first shuffled with `Reset()`and then split into mini-batches, which are then passed to the optimizer. The forward propagation calculates the cross entropy loss. The gradients of the parameters are obtained by backward propagation. 

```{.cs}
for (var e = 1; e <= epochs; ++e)
{
    batcher.Reset();

    for (var i = 1; i <= MNIST.NumTrain / batchSize; ++i)
    {
        batcher.Next(batchSize, opt, model.Images, model.Labels);
        opt.Forward();
        opt.Backward();
        opt.Optimize();
    }
}
```

The `Optimize` method updates all parameters with a stochastic gradient descent 

```{.cs}
public override void Optimize()
{
    foreach (var data in Data.Values)
    {
        if (data.Variable.Type == VariableType.Parameter)
        {
            var w = data.TensorAsExpr;
            var g = data.GradientAsExpr;
            Context.Assign(data.TensorAsValue, w - LearningRate.AsScalar(w.DataType)*g);
        }
    }
}
```

A full project can be obtained in the [sample gallery](/samples/mnist.html).

#### Multilayer Perceptron

We can improve the model by adding additional fully connected layers:

```{.cs} 
public static Model MultiLayerPerceptronModel()
{
    var images = Variable<float>(PartialShape.Create(-1, 28 * 28));
    var labels = Variable<float>(PartialShape.Create(-1, 10));
    var fc1 = new FullyConnected<float>(images, 128);
    var act1 = new ActivationReLU<float>(fc1.Output);
    var fc2 = new FullyConnected<float>(act1.Output, 64);
    var act2 = new ActivationReLU<float>(fc2.Output);
    var fc3 = new FullyConnected<float>(act2.Output, 10);

    return new Model() { Loss = new SoftmaxCrossEntropy<float>(fc3.Output, labels), Images = images, Labels = labels };
}
```

The above model training code can be reused for this more general model.


#### Convolutional Neural Nets

Convolutional neural nets exploit translation invariance and lead to a more effective representations since they are only sensitive to local information.
They are particularly suited to image classification. We define convolutional neural net with two-dimensional convolution and pooling:

```{.cs} 
public static Model ConvolutionalNeuralNetworkModel()
{
    var images = Variable<float>(PartialShape.Create(-1, 1, 28, 28));
    var labels = Variable<float>();

    var conv1 = new Convolution2D<float>(images, 5, 5, 20);
    var act1 = new ActivationTanh<float>(conv1.Output);
    var pool1 = new Pooling2D<float>(act1.Output, PoolingMode.MAX, 2, 2, 2, 2);

    var conv2 = new Convolution2D<float>(pool1.Output, 5, 5, 50);
    var act2 = new ActivationTanh<float>(conv2.Output);
    var pool2 = new Pooling2D<float>(act2.Output, PoolingMode.MAX, 2, 2, 2, 2);

    var fc1 = new FullyConnected<float>(pool2.Output, 500);
    var act3 = new ActivationTanh<float>(fc1.Output);
    var fc2 = new FullyConnected<float>(act3.Output, 10);

    return new Model() { Loss = new SoftmaxCrossEntropy<float>(fc2.Output, labels), Images = images, Labels = labels };
}
```