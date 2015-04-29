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
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);

    void evaluate(const std::vector<double>& input, std::vector<double>& output) const;
    void reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing) const;

private:    
    void train(int epochs, 
                double learningRate, 
                std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
                Node* hiddenLayer);
    
    int hiddenLayerSize;

    double** weightsFromInputLayer;
    double** weightsFromHiddenLayer;

    Node* hiddenLayer;
};
