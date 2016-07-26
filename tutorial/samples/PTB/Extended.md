***

### Natural Language Modelling

In this tutorial we implement a probabilistic model based on a recurrent neural network to predict the next work in a sentence, given the history of previous words.
To train the model we use the the [Penn Treebank dataset](http://www.cis.upenn.edu/~treebank). We largely follow the [paper](http://arxiv.org/abs/1409.2329) of 
[@Zaremba-Sutskever-Vinyals:2015].

### Regularized LSTM Networks

[@Zaremba-Sutskever-Vinyals:2015] show how to apply dropout regularization to recurrent neural networks in order to reduce overfitting. 


### LSTM with cuDNN

The `RNN` layer uses cuDNN to accelerate the forward and backward pass of multilayer LSTM networks.  

### References