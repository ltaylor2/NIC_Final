//
//  Ant.cpp
//  
//
//  Created by Daniel J. Cohen on 4/22/15.
//
//

#include "Ants.h"

#include <iostream>
#include <math.h>

Ants::Ants(int numAnts_, double evaporationFactor_, double alpha_, double beta_, Net net) :
    numAnts(numAnts_),
    evaporationFactor(evaporationFactor_),
    alpha(alpha_),
    beta(beta_),
    inputSize(net.getInputSize()),
    hiddenSize(net.getHiddenSize()),
    outputSize(net.getOutputSize())
{
    
    pheromoneFromInputLayer = new double*[inputSize];
    tourFromInputLayer = new bool *[inputSize];
    for (int i = 0; i < inputSize; i++) {
        pheromoneFromInputLayer[i] = new double[hiddenSize];
        tourFromInputLayer[i] = new bool[hiddenSize];
    }
    
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            pheromoneFromInputLayer[i][j] = static_cast<double>(rand()) / RAND_MAX;
            tourFromInputLayer[i][j] = false;
        }
    }
    
    pheromoneFromHiddenLayer = new double*[hiddenSize];
    tourFromHiddenLayer = new bool*[hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        pheromoneFromHiddenLayer[i] = new double[outputSize];
        tourFromHiddenLayer[i] = new bool[outputSize];
    }
    
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            pheromoneFromHiddenLayer[i][j] = static_cast<double>(rand()) / RAND_MAX;
            tourFromHiddenLayer[i][j] = false;
        }
    }

    // TODO get and store heuristics
    // std::vector<std::vector<double>> heuristicsInput, std::vector<std::vector<double>> heuristicsHidden                                             
}

void Ants::run(int numIterations, bool** inputStructure, bool** hiddenStructure, std::pair<std::vector<double>, std::vector<double>>>& training, std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    for (int i = 0; i < numIterations; i++) {
        createNetworkStructure();
    }
    for (int n = 0; n < inputSize; n++) {
        for (int h = 0; h < hiddenSize; h++) {
            for (int p = 0; p < outputSize; p++) {
                hiddenStructure[h][p] = bestHiddenStructure[h][p];
            }
            inputStructure[n][h] = bestInputStructure[n][h];
        }
    }
}

// TODO: How are heuristics being passed? Currently there are 2 2D vectors
// one for the input-hidden edges and one for the hidden-output edges
void Ants::createNetworkStructure() {
    double inputDenom = getInputDenom(heuristicsInput);
    double hiddenDenom = getHiddenDenom(heuristicsHidden);
    for (int a = 0; a < numAnts; a++) {
        for (int i = 0; i < inputSize; i++) {
            double randNum = static_cast<double>(rand()) / RAND_MAX;
            for (int h = 0; h < hiddenSize; h++) {
                if (getProbabilityInput(i, h, inputDenom, heuristicsInput[i][h]) > randNum) {
                    tourFromInputLayer[i][h] = true;
                    randNum = static_cast<double>(rand()) / RAND_MAX;
                    for (int o = 0; o < outputSize; o++) {
                        if (getProbabilityHidden(h, o, hiddenDenom, heuristicsHidden[h][o]) > randNum) {
                            tourFromHiddenLayer[h][o] = true;
                        }
                    }
                }
            }
        }
        // TODO this
        // add new network to vector of networks for the iteration
        // train the networks
        // find the best one and keep track of it
    }
}

// TODO do
void Ants::updatePheromones() {
    for (unsigned int i = 0; i < networks.size(); i++) {
        int numNodes = networks[i].size();
    }
}

// TODO: Check if actually returning a double
double Ants::getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic) {
    return (pow(pheromoneFromInputLayer[inputNode][hiddenNode], alpha) * pow(heuristic, beta))/denom;
}

double Ants::getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic) {
    return (pow(pheromoneFromHiddenLayer[hiddenNode][outputNode], alpha) * pow(heuristic, beta))/denom;
}

double Ants::getInputDenom(std::vector<std::vector<double>> heuristics) {
    double total = 0.0;
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            total += pow(pheromoneFromInputLayer[i][j], alpha) * pow(heuristics[i][j], beta);
        }
    }
    return total;
}

double Ants::getHiddenDenom(std::vector<std::vector<double>> heuristics) {
    double total = 0.0;
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            total += pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(heuristics[i][j], beta);
        }
    }
    return total;
}

