## Machine Learning Tools

### Network Layer Types
***

#### Fully Connected Layer

A fully connected layer connects all neurons from the previous layer with every single neuron. The output of a fully connected layer is 

$$
	E_{mu} = \sum_{v=1}^l D_{mv} W_{uv} + b_u,
$$ {#eq:convolution1}

where $W_{uv}$ are the weights and $b_u$ the bias vector. If the input tensor $D_{mv}$ has rank greater than 2, the dimensions 2 and higher are flattened. 

#### Activation Functions

Different commonly used activation functions to introduce non-linearities are:

 - Rectified linear unit $\mathrm{ReLU}(x) = \max(0, x)$
 - Sigmoid function $\sigma(x) = \dfrac{1}{1 + e^{-x}}$
 - Tangens hyperbolicus $\tanh(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$

#### Convolutional Layer

A convolutional neural net layer applies $k$ filters of size $(2w_x+1, 2w_y+1)$ to an input of $l$ features producing an output of $k$ features. The filter weight $W_{uvxy}$ connects the input feature $v$ with output feature $u$ at offset $x, y$ from a pixel. A convolution operation takes the following input

 - 4 dimensional tensor $D$ of a mini-batch of images of shape $(n, l, h, w)$ with $n$ the mini-batch size, $l$ number of input feature maps, $h$ the image height and $w$ image width.
 - 4 dimensional tensor $W$ representing the weight matrix W of shape $(k, l, 2w_x+1, 2w_y+1)$ with $k$ the number of output features, $l$ the number of input features, filter height $2w_x+1$, and filter width $2w_y+1$.

The output $E$ of a convolution operation is calculated as

$$
	E_{mu} = \sum_{v=1}^l D_{mv} * W_{uv},
$$ {#eq:convolution1}

where $*$ is the convolution operator so that

$$
	E_{muij} = \sum_{v=1}^l \sum_{x=-w_x}^{w_x} \sum_{y=-w_y}^{w_y} D_{m,v,i+x,j+y} W_{uvxy}.
$$ {#eq:convolution2}

#### Pooling Layer

The purpose of a pooling layer is to control overfitting by reducing the spatial size of the representation hence reducing the amount of parameters and computation in the network. The pooling parameters are a windows size $f$ and a stride $s$. A mini batch of size $(n, c, h, w)$ is transformed to a mini batch of size $(n, c, (h - f)/s + 1, (w - f)/s + 1)$. The max pooling tranforms an input tenser is then given by

$$
	E_{muij} = \max \{ D_{m,u,is + x, js + y} \mid x,y = 0, \ldots f-1 \}.
$$ {#eq:pooling}

Alternative pooling operations are average pooling or $L_2$-norm pooling. Usually $h - f$ and $w - f$ are multiples of the stride $s$. If this is not the case the output size of pooling layer is the floor of $(h - f)/s$ respectively $(w - f)/s$.

#### Softmax

The softmax function is a normalized exponential function. It takes a $K$ dimensional vector $z$ and outputs a $K$ dimensional vector of real values in $[0,1]$ 

$$
\mathrm{softmax}(z)_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}.
$$ {#eq:softmax}

The softmax function generalizes the logistic function to output a multiclass categorical probability distribution.
To calculate the softmax function we use the following observation. With $m = \max_j z_j$ we can calculate the 
logarithm of the softmax  

$$
\begin{align}
	\log(\mathrm{softmax}(z)_k) &= z_k - \log\left(\sum_{j=1}^K \exp(z_j + m - m)\right) \\
		&= z_k - \log\left(\exp(m) \sum_{j=1}^K \exp(z_j - m)\right) \\
		&= z_k - m - \log\left(\sum_{j=1}^K \exp(z_j - m)\right),
\end{align}
$$ {#eq:log-softmax}

which is more stable because $\exp(z_j - m)$ of the largest $z_j$ is mapped to one. This leads to the following stable formula to 
calculate the softmax function:

$$
	\mathrm{softmax}(z)_k = \exp\left(z_k - m - \log\left(\sum_{j=1}^K \exp(z_j - m)\right)\right).
$$ {#eq:stable-softmax}

#### Softmax with Cross Entroy

The cross entropy quantifies the difference of two probability distributions over the same random variable. For discrete probability distributions it is defined as

$$
	\mathrm{D}(p \parallel q) = - \sum_k p_k \log(q_k).
$$ {#eq:cross-entropy}

The input to a softmax with cross entropy layer are

- Class index labels $k_m$  
- Values $z_m$ from which probabilities $q_{mk} = \mathrm{softmax}(z_m)_k$ are calculated according to ({@eq:stable-softmax})

The output is the

$$
	L = - \sum_m \sum_k \log(q_{m k_m}) \in \mathbb{R}.
$$ {#eq:softmax-cross-entropy}

In order to relate ({@eq:cross-entropy}) and ({@eq:softmax-cross-entropy}) take $p_k$ to be the 1-hot vector $p_{mk} = \delta_{k,k_m}$ with $m$ the sample index and $k$ the class index.

### Optimization Algorithms
***











