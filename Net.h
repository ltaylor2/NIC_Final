#pragma once

#include <vector>
#include <utility>

#include "Node.h"

class Net {
public:
    Net(int epochs_, 
        int hiddenLayerSize_,
        double learningRate_, 
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
        bool** inputStructure,
        bool** hiddenStructure);

    void evaluate(const std::vector<double>& input, std::vector<double>& output) const;
    double reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);

    int getInputSize() { return inputSize; }
    int getHiddenLayerSize() { return hiddenLayerSize; }
    int getOutputSize() { return outputSize; }
    double getTotalError() { return totalError; }

    bool getInputEdge(int i, int h) { return inputStructure[i][h]; }
    bool getHiddenEdge(int h, int o) { return hiddenStructure[h][o]; }

    int getEpochs() { return epochs; }
    double getLearningRate() { return learningRate; }

private:    
    void train(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
                Node* hiddenLayer);
    
    int epochs;

    int inputSize;
    int hiddenLayerSize;
    int outputSize;

    double learningRate;

    double** weightsFromInputLayer;
    double** weightsFromHiddenLayer;

    Node* hiddenLayer;

    bool** inputStructure;
    bool** hiddenStructure;

    double totalError;
};
