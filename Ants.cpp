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
#include <limits>

Ants::Ants(int numAnts_, double evaporationFactor_, double alpha_, double beta_, Net net) :
    numAnts(numAnts_),
    evaporationFactor(evaporationFactor_),
    alpha(alpha_),
    beta(beta_),
    inputSize(net.getInputSize()),
    hiddenSize(net.getHiddenLayerSize()),
    outputSize(net.getOutputSize()),
    originalNet(net),
    bestNet(net)
{
    

    pheromoneFromInputLayer = new double*[inputSize];
    tourFromInputLayer = new bool*[inputSize];
    bestInputStructure = new bool*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        pheromoneFromInputLayer[i] = new double[hiddenSize];
        tourFromInputLayer[i] = new bool[hiddenSize];
        bestInputStructure[i] = new bool[hiddenSize];
    }
    
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            pheromoneFromInputLayer[i][j] = static_cast<double>(rand()) / RAND_MAX;
            tourFromInputLayer[i][j] = false;
            bestInputStructure[i][j] = false;
        }
    }

    pheromoneFromHiddenLayer = new double*[hiddenSize];
    tourFromHiddenLayer = new bool*[hiddenSize];
    bestHiddenStructure = new bool*[hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        pheromoneFromHiddenLayer[i] = new double[outputSize];
        tourFromHiddenLayer[i] = new bool[outputSize];
        bestHiddenStructure[i] = new bool[outputSize];
    }
    
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            pheromoneFromHiddenLayer[i][j] = static_cast<double>(rand()) / RAND_MAX;
            tourFromHiddenLayer[i][j] = false;
            bestHiddenStructure[i][j] = false;
        }
    }

    std::cout << "eeeeee" << std::endl;
    // TODO get and store heuristics   
    bestError = std::numeric_limits<double>::max();      
    heuristic = bestError;                            
}

void Ants::run(int numIterations, bool** inputStructure, bool** hiddenStructure, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    for (int i = 0; i < numIterations; i++) {
        createNetworkStructure(training, testing);
        updatePheromones(bestNet.getTotalError());
    }

    for (int n = 0; n < inputSize; n++) {
        for (int h = 0; h < hiddenSize; h++) {
            for (int p = 0; p < outputSize; p++) {
                hiddenStructure[h][p] = bestNet.getHiddenEdge(h, p);
            }
            inputStructure[n][h] = bestNet.getInputEdge(n,h);
        }
    }
}

// TODO: How are heuristics being passed? Currently there are 2 2D vectors
// one for the input-hidden edges and one for the hidden-output edges
void Ants::createNetworkStructure(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                  std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    double inputDenom = getInputDenom(heuristic);
    double hiddenDenom = getHiddenDenom(heuristic);
    
    for (int a = 0; a < numAnts; a++) {
        // initialize space for network TODO: check if actually auto-initializes to false/0?
        bool** tourFromInputLayer = new bool*[inputSize];
        bool** tourFromHiddenLayer = new bool*[hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            tourFromInputLayer[i] = new bool[hiddenSize]();
        }
        for (int i = 0; i < hiddenSize; i++) {
            tourFromHiddenLayer[i] = new bool[outputSize]();
        }
        
        for (int i = 0; i < inputSize; i++) {
            double randNum = static_cast<double>(rand()) / RAND_MAX;
            for (int h = 0; h < hiddenSize; h++) {
                if (getProbabilityInput(i, h, inputDenom, heuristic) > randNum) {
                    tourFromInputLayer[i][h] = true;
                    randNum = static_cast<double>(rand()) / RAND_MAX;
                    for (int o = 0; o < outputSize; o++) {
                        if (getProbabilityHidden(h, o, hiddenDenom, heuristic) > randNum) {
                            tourFromHiddenLayer[h][o] = true;
                        }
                    }
                }
            }
        }
        // add new network to vector of networks for the iteration
        Net newNet(originalNet.getEpochs(), hiddenSize, originalNet.getLearningRate(), training, tourFromInputLayer, tourFromHiddenLayer);
        networks.push_back(newNet);
    }

    // test all the networks and find the best one
    for (unsigned int i = 0; i < networks.size(); i++) {
        double currError = networks[i].reportErrorOnTestingSet(testing);
        if (currError < bestError) {
            bestError = currError;
            bestNet = networks[i];
        }
    }
}

void Ants::updatePheromones(double error) {
    for (int i = 0; i < inputSize; i++) {
        for (int h = 0; h < hiddenSize; h++) {
            for (int o = 0; o < outputSize; o++) {
                pheromoneFromHiddenLayer[h][o] = (1 - evaporationFactor) * pheromoneFromHiddenLayer[h][o];
                if (bestNet.getHiddenEdge(h, o) == 1) {
                    pheromoneFromHiddenLayer[h][o] += (evaporationFactor * (1/error));
                }
            }
            pheromoneFromInputLayer[i][h] = (1 - evaporationFactor) * pheromoneFromInputLayer[i][h];
            if (bestNet.getInputEdge(i, h) == 1) {
                pheromoneFromInputLayer[i][h] += (evaporationFactor * (1/error));
            }
        }
    }
}

// TODO: Check if actually returning a double
double Ants::getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic) {
    return (pow(pheromoneFromInputLayer[inputNode][hiddenNode], alpha) * pow(heuristic, beta))/denom;
}

// TODO heuristic
double Ants::getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic) {
    return (pow(pheromoneFromHiddenLayer[hiddenNode][outputNode], alpha) * pow(heuristic, beta))/denom;
}

// TODO heuristic
double Ants::getInputDenom(double heuristic) {
    double total = 0.0;
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            total += pow(pheromoneFromInputLayer[i][j], alpha) * pow(heuristic, beta);
        }
    }
    return total;
}

// TODO 2d heuristics to calculate actual value
double Ants::getHiddenDenom(double heuristic) {
    double total = 0.0;
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            total += pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(heuristic, beta);
        }
    }
    return total;
}

