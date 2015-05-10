//
//  Ant.h
//  
//
//  Created by Daniel J. Cohen on 4/22/15.
//
//

#pragma once

#include "Node.h"
#include "Net.h"

#include <queue>
#include <vector>

class Ants {
public:
    Ants(int numAnts_, double evaporationFactor_, double alpha_, double beta_, Net net);
    void run(int numIterations,
                   bool** inputStructure,
                   bool** hiddenStructure,
                   std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                   std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);

    void createNetworkStructure(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);
    double getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic);
    double getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic);
    double getInputDenom(double heuristic);  //TODO 2d
    double getHiddenDenom(double heuristic); //TODO 2d
    void updatePheromones(double error);

private:
    int numAnts;
    double evaporationFactor, alpha, beta;

    int inputSize, hiddenSize, outputSize;

    double** pheromoneFromInputLayer;
    double** pheromoneFromHiddenLayer;

    bool** bestInputStructure;
    bool** bestHiddenStructure;
    
    Net originalNet;
    
    double bestError;
    Net bestNet;
    
    double heuristic;
    
    std::vector<Net> networks;
};

