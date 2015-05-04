//
//  Ant.h
//  
//
//  Created by Daniel J. Cohen on 4/22/15.
//
//

#pragma once

#include "Node.h"

#include <queue>
#include <vector>

class Ants {
public:
    Ants(int numAnts_, double evaporationFactor_, double alpha_, double beta_, int inputSize_, int hiddenSize_, int outputSize_);
    void Ants::run(int numIterations,
                   bool** inputStructure,
                   bool** hiddenStructure,
                   std::pair<std::vector<double>, std::vector<double>>>& training,
                   std::pair<std::vector<double>, std::vector<double>>>& testing);

    void createNetworkStructure();
    double getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic);
    double getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic);
    double getInputDenom(std::vector<std::vector<double>> heuristics);
    double getHiddenDenom(std::vector<std::vector<double>> heuristics);
    void updatePheromones();

private:
    int numAnts;
    double evaporationFactor, alpha, beta;

    int inputSize, hiddenSize, outputSize;

    std::vector<std::queue<int>> networks;

    double** pheromoneFromInputLayer;
    double** pheromoneFromHiddenLayer;

    bool** tourFromInputLayer;
    bool** tourFromHiddenLayer;
};

