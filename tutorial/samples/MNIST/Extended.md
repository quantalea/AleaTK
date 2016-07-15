***

### Multinomial Logistic Regression

The [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) model is the most simple classifier for categorical dependent variables. It represents the probability of the dependent variable $Y$ to be of class $k$ 

$$
P(Y=k \mid x, W, b) = \frac{\exp(W_k x + b_k)}{\sum_{j=1}^K \exp(W_j x + b_j)} = \mathrm{softmax}(W x + b)_k
$$ {#eq:mnl-regression}

where $\mathrm{softmax}(W x + b)$ is the softmax function with $W \in \mathbb{R}^{K \times D}$ the weight matrix with rows $W_k$, $b \in \mathbb{R}^K$ the bias vector and $x \in \mathbb{R}^D$ the data or explanatory variable. We use the cross entropy 

$$
	\mathrm{D}(p \parallel q) = - \sum_k p_k \log(q_k)
$$ {#eq:cross-entropy}

to measure the discrepancy between predicted probabilities ({@eq:mnl-regression}) and the actual class of a sample. 


### Multilayer Perceptron

We extend ({@eq:mnl-regression}) by adding 3 layers with a ReLU activation function so that

$$
P(Y=k \mid x, W_1, W_2, W_3) = \mathrm{softmax}(\mathrm{ReLu}(W_3 \; \mathrm{ReLu}(W_2 \; \mathrm{ReLu}(W_1 x))))_k
$$ {#eq:mlp-regression}


### Convolutional Neural Net

Convolutional neural nets consist of multiple layers of small neuron collections, called receptive fields. They process portions of the input image. The outputs of these collections are tiled so that their input regions overlap. Tiling adds translation invariance and leads to a better representation of the input image. Convolutional neural nets usually contain pooling layers, which perform a nonlinear down-sampling to reduce overfitting.

More details on convolutional neural nets can be found [here](https://en.wikipedia.org/wiki/Convolutional_neural_network), [here](http://cs231n.github.io/convolutional-networks/), [here](http://deeplearning.net/tutorial/lenet.html) and [here](http://cs.stanford.edu/people/karpathy/convnetjs/).

We imporve model ({@eq:mnl-regression}) with the following convolutional neural network architecture: 

 - Convolutional layer of windows size 5 with 20 output features followed by $tanh$ nonlinearity 
 - A max pooling with windows size 2 and stride 2
 - Convolutional layer of windows size 5 with 50 output features followed by $tanh$ nonlinearity 
 - A max pooling with windows size 2 and stride 2
 - A fully connected layer with 500 neurons followed by $tanh$ nonlinearity
 - A fully connected layer to map to 10 dimensions for the 10 categories
 - A softmax layer to map the 10 dimensions to categorical probabilities
 
