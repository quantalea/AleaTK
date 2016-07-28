***

### Natural Language Modelling

In this tutorial we implement a probabilistic model based on a recurrent neural network to predict the next work in a sentence, given the history of previous words. 

We use the the [Penn Treebank dataset](http://www.cis.upenn.edu/~treebank). It has a vocabulary of 10k words and consists of 929k training words, 73k validation words, and 82k test words. We download a preprocessed version from Tomáš Mikolov's [web page](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz). We follow the approach proposed in @DBLP:journals/corr/ZarembaSV14 [section 4.1].

The preprocessed input data is a sequence of words containing `<unk>` for words not in the vocabulary, `<eos>` to mark the end of the sequence and `N` to represent numbers. We translate the sequence of words into a sequence of integers, representing the indices of the words in the vocabulary of size $s$. We create minibatches of size $b$ by taking consecutively $b$ subsequences of fixed length $n$ from this sequence. Taking fixed length subsequences would possibly break up a sentence into multiple subsequences. As we are mainly concerned about the history of words to predict the next word, this simplification is an acceptable compromise and it allows us to unroll the LSTM layers by a fixed amount of steps.  

 - Input data: integer matrix $x \in \mathbb{N}_0^{n \times b}$, where $n$ is the sequence length and $b$ the minibatch size
 - Embedding layer: translating $x$ into $y \in \mathbb{R}^{n \times b \times d}$ where $d$ is the dimension of the embedding, defined in terms of the embedding weights $w \in \mathbb{R}^{s, d}$ by $y_{i,j,k} = w_{x_{i,j},k}$ 
 - Two [LSTM layers](/ml_tools.html#eq:lstm2) with input and hidden dimension $d$, unrolled $n$ steps along the first dimension of $y$, producing output $z \in \mathbb{R}^{n \times b \times d}$
 - [Fully connected layer](/ml_tools.html#eq:fully-connected): the input is $z$ reshaped to a matrix $\mathbb{R}^{n b \times d}$ and transformed to the output $u \in \mathbb{R}^{n b \times s}$, with $s$ the size of the vocabulary
 - [Softmax with cross entropy layer](/ml_tools.html#eq:softmax-cross-entropy): transforms $u$ into probabilities $p \in \mathbb{R}^{n b \times s}$ and calculates cross entropy $D(p \| q)$ from labels $q \in \mathbb{N}_0^{n \times b}$, given by the index in the vocabulary of the next word for each word in $x$, reshaped to an element of $\mathbb{N}_0^{n b}$.
 
The hidden states of the LSTM layers are initialized to zero for the first minibatch. As we traverse the sequence sequentially with minibatches of size $n*b$ we can take the final hidden states of a minibatch as the initial hidden state of the next minibatch. 

### Regularized LSTM Networks

@DBLP:journals/corr/ZarembaSV14 show how to apply dropout regularization to recurrent neural networks in order to reduce overfitting. 
They suggest to apply the dropout operator only to the non-recurrent connections, including the input and the output connections. 

### LSTM with cuDNN

The `RNN` layer uses cuDNN to accelerate the forward and backward pass of multilayer LSTM networks. The article @DBLP:journals/corr/AppleyardKB16 describes in more detail the optimizations for LSTM nodes implemented in cuDNN. 

***

### References