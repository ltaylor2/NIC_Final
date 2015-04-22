#pragma once

#include <vector>
#include <utility>

#include "Node.h"

// Class for representing and training a perceptron network
class Net {
public:
    // Constructor
    // @param epochs, number of iterations training on training set
    // @param learningRate, learning rate used in train, affects speed of training and convergence guarentees
    // @param multipleOutputs, true for 10 output node representation, false for 1
    // @param training, the training set, input and desired output
    // @param testing, the testing set, input and desired output
    Net(int epochs, 
        int hiddenSize_,
        double learningRate, 
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);
    // Evaluates perceptron on some input and stores output in output param
    // @param input, the input to run the network on
    // @param output, a vector to fill with the output of the calculation
    // NOTE assumes output is correct size
    void evaluate(const std::vector<double>& input, std::vector<double>& output) const;

private:
    // Trains the perceptron, helper method called by constructor
    // @param epochs, number of iterations training on training set
    // @param learningRate, learning rate used in train, affects speed of training and convergence guarentees
    // @param multipleOutputs, true for 10 output node representation, false for 1
    // @param training, the training set, input and desired output
    // @param testing, the testing set, input and desired output
    // void train(int epochs, 
    //            double learningRate, 
    //            bool multipleOutputs,
    //            std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
    //            std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);
    // Reports testing set error via stdout, called every epoch during training
    // @param multipleOutputs, true for 10 output node representation, false for 1
    // @param testing, the testing set, input and desired output
    // void reportErrorOnTestingSet(bool multipleOutputs,
    //                              std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing) const;

    // Helper method, for weight update rule
    double sigmoidPrime(double sum) const;
    
    int hiddenSize;

    double** weightsFromInputLayer;
    double** weightsFromHiddenLayer;

    Node* hiddenLayer;
};
