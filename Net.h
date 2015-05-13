#pragma once

#include <vector>
#include <utility>

#include "Node.h"

// Neural network class with one hidden layer that implements backpropogation 
// for learning
// NOTE not necessarilly fully connected, see docs of inputStructure and hiddenStructure
class Net {
public:
    // Constructor
    // @param epochs_, number of epochs of training
    // @param hiddenLayerSize_, number of hidden nodes in network
    // @param training, vector of pairs of labeled inputs and outputs
    // @param inputStructure, records which edges from input layer to hidden layer
    //                        are used in the neural network
    // @param hiddenStructure, records which edges from hidden layer to output layer
    //                         are used in the neural network
    Net(int epochs_, 
        int hiddenLayerSize_,
        double learningRate_, 
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
        bool** inputStructure,
        bool** hiddenStructure);

    // Destructor
    ~Net();

    // Evaluate training network on some input
    // @param input, input to be evaluated
    // @param output, vector is filled with output from neural network
    void evaluate(const std::vector<double>& input, std::vector<double>& output) const;

    // Run network on testing set and report error
    // @param testing, vector of pairs of labeled inputs and outputs
    // @returns percentage of examples network got correct
    double reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);

    // Getters needed by Ants for hybridization
    int getInputSize() { return inputSize; }
    int getHiddenLayerSize() { return hiddenLayerSize; }
    int getOutputSize() { return outputSize; }
    double getTotalError() { return totalError; }

    bool getInputEdge(int i, int h) { return inputStructure[i][h]; }
    bool getHiddenEdge(int h, int o) { return hiddenStructure[h][o]; }

    double** getInputHeuristics() { return inputHeuristics; }
    double** getHiddenHeuristics() { return hiddenHeuristics; }

    int getEpochs() { return epochs; }
    double getLearningRate() { return learningRate; }

private:    
    // Train network using backpropogation learning
    void train(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training); 
    
    // Single hidden layer NN parameters
    int epochs;
    int inputSize;
    int hiddenLayerSize;
    int outputSize;
    double learningRate;

    // NN weights
    double** weightsFromInputLayer;
    double** weightsFromHiddenLayer;

    // Hidden layer uses Node class to store computed output
    Node* hiddenLayer;

    // Marks which edges in NN are considered part of the network
    // NOTE thus not a fully connected network
    bool** inputStructure;
    bool** hiddenStructure;

    // Heuristic used by Ants
    // Number of times weight changes during testing
    double** inputHeuristics;
    double** hiddenHeuristics;

    // Error used by Ants to select edges
    double totalError;
};
