***

### Natural Language Modelling

In this tutorial we implement a probabilistic model based on a recurrent neural network to predict the next work in a sentence, given the history of previous words.
To train the model we use the the [Penn Treebank dataset](http://www.cis.upenn.edu/~treebank). We largely follow @DBLP:journals/corr/ZarembaSV14 [section 4.1].

### Regularized LSTM Networks

@DBLP:journals/corr/ZarembaSV14 show how to apply dropout regularization to recurrent neural networks in order to reduce overfitting. 
They suggest to apply the dropout operator only to the non-recurrent connections, including the input and the output connections. 

### LSTM with cuDNN

The `RNN` layer uses cuDNN to accelerate the forward and backward pass of multilayer LSTM networks. The article @DBLP:journals/corr/AppleyardKB16 describes in more
detail the optimizations for LSTM nodes implemented in cuDNN. 

***

### References