#pragma once

#include <vector>
#include <utility>

#include "Node.h"

class Net {
public:
    Net(int epochs, 
        int hiddenLayerSize_,
        double learningRate, 
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
        bool** inputStructure,
        bool** hiddenStructure);

    void evaluate(const std::vector<double>& input, std::vector<double>& output) const;
    void reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing) const;

    int getInputSize() { return inputSize; }
    int getHiddenLayerSize() { return hiddenLayerSize; }
    int getOutputSize() { return outputSize; }
private:    
    void train(int epochs, 
                double learningRate, 
                std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
                Node* hiddenLayer);
    
    int inputSize;
    int hiddenLayerSize;
    int outputSize;

    double** weightsFromInputLayer;
    double** weightsFromHiddenLayer;

    Node* hiddenLayer;

    bool** inputStructure;
    bool** hiddenStructure;
};
