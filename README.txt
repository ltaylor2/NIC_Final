**************************************************************************************
Determining Optimized Neural Network Structure Using Hybrid Ant Colony Optimization Approach
***************************************************************************************

Dan "Only One Nickname" Cohen
Josh "Google C++ Style Guide" Imhoff
Liam "The Scientist" Taylor
May 13, 2015
Final Project
CS3445

This project is designed to research and report on the effectiveness of hybridizing a one-hidden-layer neural network with an implementation of Ant Colony Optimization. Many parameters are hard-coded in the main method. Other can be changed from the command line (see below). This program can tests on a Connect Four data set with either a standard, fully connected neural network, or can be run with the more advanced hybrid algorithm. Output notes mark the status of the program as it runs.

**Memory Leak Note**
Due to our high runtime of ACO, particularly in combination with training and testing networks, our implementation includes numerous dynamic arrays, allocated with "new". While many of them can be safely destructed, some fundamental organization of our program prevents others from destructing appropriately. These memory leaks, however, do NOT affect program while it is running.

BUILD
    make


USAGE
    ./aconn connect4DataFile numberOfEpochsForNNLearning learningRate numberOfHiddenNodes ants [0 = false | 1 = true] numberOfAntIterations

