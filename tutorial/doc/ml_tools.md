## Machine Learning Tools

### Network Layer Types
***

#### Activation Functions

Different commonly used activation functions to introduce non-linearities are:

 - Rectified linear unit $\mathrm{ReLU}(x) = \max(0, x)$
 - Sigmoid function $\sigma(x) = \dfrac{1}{1 + e^{-x}}$
 - Hyperbolic tangent $\tanh(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$

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
	\mathrm{D}(p \| q) = - \sum_k p_k \log(q_k).
$$ {#eq:cross-entropy}

The input to a softmax with cross entropy layer are

- Class index labels $k_m$  
- Values $z_m$ from which probabilities $q_{mk} = \mathrm{softmax}(z_m)_k$ are calculated according to ({@eq:stable-softmax})

The output is the

$$
	L = - \sum_m \sum_k \log(q_{m k_m}) \in \mathbb{R}.
$$ {#eq:softmax-cross-entropy}

In order to relate ({@eq:cross-entropy}) and ({@eq:softmax-cross-entropy}) take $p_k$ to be the 1-hot vector $p_{mk} = \delta_{k,k_m}$ with $m$ the sample index and $k$ the class index.

#### Fully Connected Layer

A fully connected layer connects all neurons from the previous layer with every single neuron. The output of a fully connected layer is 

$$
	E_{mu} = \sum_{v=1}^l D_{mv} W_{uv} + b_u,
$$ {#eq:fully-connected}

where $W_{uv}$ are the weights and $b_u$ the bias vector. If the input tensor $D_{mv}$ has rank greater than 2, the dimensions 2 and higher are flattened. 

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

#### Dropout Layer

A layer randomly sets input values to zero and acts as a regularizer. More details are in the following [paper](http://www.cs.toronto.edu/~hinton/absps/dropout.pdf).

#### RNN

The standard RNN dynamics is given by a deterministic state transition function from previous to current hidden
states

$$
	\mathrm{RNN} : (x_t, h_{t-1})  \mapsto  h_{t} = f(T_{n,d} x_t + T_{n,n}h_{t-1})
$$

with $W_{n, d} : \mathbb{R}^n \rightarrow \mathbb{R}^d$ an affine map. Usually $f = \tanh$ or $f = \sigma$.

#### LSTM

The LSTM has a dynamics that allow it to keep information for an extended number of time steps. There are different LSTM architectures that differ in their connectivity structure and activation functions. All LSTM architectures have explicit memory cells  $c_t \in \mathbb{R}^d$ for storing information for long periods of time and can decide to overwrite, retrieve, or keep the memory cell for the next time step. An LSTM cell transforms an input $x_t \in \mathbb{R}^n$ and a hidden state $h_{t-1} \in \mathbb{R}^d$, keeping a cell state $c_t \in \mathbb{R}^d$ as follows: 

$$
\begin{align}
	&\mathrm{LSTM} : (x_t, h_{t-1}, c_{t-1})  \mapsto  (h_{t}, c_{t}) \\
	&\left(
		\begin{array}{c}
			i \\
			f \\
			o \\
			g 
		\end{array}			
	\right) = 
	\left(
		\begin{array}{c}
			\sigma \\
			\sigma \\
			\sigma \\
			\zeta_1 
		\end{array}			
	\right) T_{n+d, 4d} 
	\left(
		\begin{array}{c}
			x_t \\
			h_{t-1} 
		\end{array}			
	\right) \\
	&c_t = f \odot c_{t-1} + i \odot g \\
	&h_t = o \odot \zeta_2(c_t)
\end{align} 
$$ {#eq:lstm1}

with $T_{n+d, 4d} : \mathbb{R}^{n+d} \rightarrow \mathbb{R}^{4d}$ an affine map. Usually $\zeta_i = \tanh$. The vector $i$ is called the input gate, $f$ the forget gate, $o$ the output gate and $g$ the input modulation gate.

Often multiple LSTM layers are stacked. In this case $d = n$ excpet for the first layer, where the input dimension $n$ can be different from the hidden dimension $d$.

LSTMs are regularized with dropout. For stacked LSTMs the dropout operator $D$ is only applied to non-recurrent connections between layers, to the input of the first layer and to the output of the last layer:

$$
\begin{align}
	&\mathrm{LSTM} : (h^{l-1}_t, h^l_{t-1}, c^l_{t-1})  \mapsto  (h^l_{t}, c^l_{t}) \\
	&\left(
		\begin{array}{c}
			i \\
			f \\
			o \\
			g 
		\end{array}			
	\right) = 
	\left(
		\begin{array}{c}
			\sigma \\
			\sigma \\
			\sigma \\
			\zeta_1 
		\end{array}			
	\right) T_{2d, 4d} 
	\left(
		\begin{array}{c}
			D(h^{l-1}_t) \\
			h^l_{t-1} 
		\end{array}			
	\right) \\
	&c^l_t = f \odot c^l_{t-1} + i \odot g \\
	&h^l_t = o \odot \zeta_2(c^l_t)
\end{align} 
$$ {#eq:lstm2}

Adding $D$ as well to the hidden state and the cell state flowing from one cell to the next would corrupt the information flow and deteriorate the LSTMs ability to learn long term events. Details are in this [paper](https://arxiv.org/pdf/1409.2329.pdf).

The forward pass is fully specified by ({@eq:lstm1}), ({@eq:lstm2}). For the backward path let $L$ be a loss function and let $dh_t = \partial L / \partial h_t$. We start with

\begin{align}
	& do_k \equiv \frac{\partial L}{\partial o_k} = \sum_j \frac{\partial L}{\partial h_{t,j}} \frac{\partial h_{t,j}}{\partial o_k} = 
		dh_{t,k} \cdot \zeta_2(c_{t,k}) \quad	 
	\Rightarrow \quad do = dh_t \odot \zeta_2(c_t)	
\end{align}

Because $c_t$ is kept as an internal state and updated we have 

\begin{align}
	& dc_{t,k} \equiv \frac{\partial L}{\partial c_{t,k}} = dh_{t,k} \cdot o_k \cdot \zeta_2'(c_{t,k}) \quad	 
	\Rightarrow \quad dc_t = dc_t + dh_t \odot o \odot \zeta_2'(c_t)	
\end{align}

From $c_t = f \odot c_{t-1} + i \odot g$ we find

\begin{align}
	& di_k \equiv \frac{\partial L}{\partial i_k} = \sum_j \frac{\partial L}{\partial c_{t,j}} \frac{\partial c_{t,j}}{\partial i_k} =
		dc_{t,k} \cdot g_k \quad	 
	\Rightarrow \quad di = dc_t \odot g
\end{align}

\begin{align}
	& dg_k \equiv \frac{\partial L}{\partial g_k} = dc_{t,k} \cdot i_k \quad	 
	\Rightarrow \quad dg = dc_t \odot i	
\end{align}

\begin{align}
	& df_k \equiv \frac{\partial L}{\partial f_k} = dc_{t,k} \cdot c_{t-1,k} \quad	 
	\Rightarrow \quad df = dc_t \odot c_{t-1}	
\end{align}

\begin{align}
	& dc_{t-1,k} \equiv \frac{\partial L}{\partial c_{t-1,k}} = dc_{t,k} \cdot f_k \quad
	\Rightarrow \quad dc_{t-1} = dc_t \odot f	
\end{align}

With $(i, f, o, g) = (\sigma(\hat i), \sigma(\hat f), \sigma(\hat o), \zeta_1(\hat g))^{\top}$

\begin{align}
	& d\hat{i}_k \equiv \frac{\partial L}{\partial \hat{i}_k} = di_k \cdot i_k \cdot (1 - i_k) \quad
	\Rightarrow \quad d\hat{i} = di \odot i \odot (1 - i)	
\end{align}

\begin{align}
	& d\hat{f}_k \equiv \frac{\partial L}{\partial \hat{f}_k} = df_k \cdot f_k \cdot (1 - f_k) \quad
	\Rightarrow \quad d\hat{f} = df \odot f \odot (1 - f)	
\end{align}

\begin{align}
	& d\hat{o}_k \equiv \frac{\partial L}{\partial \hat{o}_k} = do_k \cdot o_k \cdot (1 - o_k) \quad
	\Rightarrow \quad d\hat{o} = do \odot o \odot (1 - o)	
\end{align}

\begin{align}
	& d\hat{g}_k \equiv \frac{\partial L}{\partial \hat{g}_k} = dg_k \cdot \zeta_1'(g_k) \quad
	\Rightarrow \quad d\hat{g} = dg \odot \zeta_1'(g)	
\end{align}

With $\hat{z}_t = (\sigma(\hat i), \sigma(\hat f), \sigma(\hat o), \zeta_1(\hat g))^{\top}$, $(x_t, h_{t-1})^{\top} = z_t$ and 
$\hat{z}_t = T_{n+d, 4n} z_t$ 

\begin{align}
	& dz_t \equiv \frac{\partial L}{\partial z_t} = T_{n+d, 4n}^{\top} d\hat{z}_t
\end{align}

from which we obtain $dz_t = (dx_t, dh_{t-1})^{\top}$ and for the gradient $dT_{n+d, 4n}$ of weight parameters is obtained as the outer product 

\begin{align}
	dT_{n+d, 4n} = d\hat{z}_t \otimes dz_t
\end{align}

A more detailed derivations can be found [here](http://arunmallya.github.io/writeups/nn/lstm/index.html) and in the supplementary material of this [paper](https://arxiv.org/pdf/1503.04069v1.pdf).


### Optimization Algorithms
***

#### Stochastic Gradient Descent 

#### RMSProp 









