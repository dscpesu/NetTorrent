# NetTorrent - DSC PESU

NetTorrent is a project whose aim is to completely decentralise the training of a neural network by using data parallelism along with model parallelism. 

The primary motive is to speed up training time while achieving an accuracy close to the ones obtained by conventional centralised and distributed training. 

Let n be the number of layers in the network being trained. NetTorrent will then have n peer-to-peer networks (p2p networks), each network responsible for a single layer. 

Within each p2p network consider m nodes in the network, among m nodes of the network the workload for training is divided by dividing the number of neurons amongst the peers. This implements Each p2p network receives a copy of the entire training data, and divides the data amongst the m nodes of the network. In this manner we achieve data parallelism in hopes of cutting down the training time. 

After the completion of training of the (n-1)th p2p network, the gradients are passed to the nth network, and the above process is repeated. A feed forward network would implement asynchronous gradient descent to update the parameters amongst the different p2p networks. 


## Implementation: 
Consider a network with 1000 users and a training dataset having 10000 samples.
The entire network is divided into 10 different p2p networks with each p2p network having 100 users and the entire dataset is also divided into 10 different sets with each set having 1000 samples.

## (i) Data Parallelism 

### (For a single layer)
Set-1 is assigned to the first 10 user, Set-2 to the next 10 users within the first p2p network and so on.

Each node(user) in the p2p network  trains for n samples(say n = 100) and then the one with the best results exchanges info about the weights with the rest of the nodes and this process is repeated till a threshold accuracy is reached which is decided by the user.

### (For a network of 10 layers)
The above process is repeated in p2p-network-2, p2p-network-3, … ,p2p-network-10 for 10 different layers.

By using data parallelism, a better accuracy can be achieved because only the best model is considered after each iteration.

## (ii) Model Parallelism(Mini-batch gradient descent)
The p2p networks from the data parallelism also use model parallelism along with data parallelism.

Each p2p network has 100 users which corresponds to a single layer.Suppose layer 1 has 200 neurons and layer 2 has 50 neurons and input layer consists of 100 samples with 50 rows of data.

### a)Forward Pass
During the forward pass in the first layer, we have to multiply a 100x50 tensor with a 50x200 matrix.

The first matrix is divided into 10 parts which results in 10 different 10x50 matrices.
These are multiplied separately with the 50x200 matrix corresponding to the weights resulting in the input matrix for the next layer

### b)Backward Pass
During the backward pass, the same process is repeated as forward pass but is subdivided again into 2 parts : one gradient is used to compute the gradients for weights and one more is used to send to the (n-1)th layer.
Since, majority of the training process involves tensor computation and by dividing this workload among nodes reduces the training time.

The above example was with respect to mini batch gradient descent where at each layer only 3 tensor computations are required.When using momentum gradient descent or adaptive optimizers like adaGrad, RMSProp or Adam which require 5-10 tensor computations at each layer, this method can reduce the workload and training time without little or no trade off in accuracy.
