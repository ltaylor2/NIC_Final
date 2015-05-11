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
    inputHeuristics(net.getInputHeuristics()),
    hiddenHeuristics(net.getHiddenHeuristics()),
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
        for (int j = 0; j < hiddenSize; j++) {
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
}

void Ants::run(int numIterations, bool** inputStructure, bool** hiddenStructure, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    for (int i = 0; i < numIterations; i++) {
                //std::cout << "a" << std::endl;

        createNetworkStructure(training, testing);
                //std::cout << "b" << std::endl;
        std::cout << "Updating pheromones." << std::endl;
        updatePheromones(bestNet.getTotalError());
        std::cout << "Completed Ant iteration " << i <<  std::endl;

    }

                        //std::cout << "d" << std::endl;

    std::cout << "Returning best ant structure." << std::endl;
    for (int i = 0; i < inputSize - 1; i++) {
        for (int h = 0; h < hiddenSize; h++) {
            inputStructure[i][h] = bestNet.getInputEdge(i, h);
            if (inputStructure[i][h])
                std::cout << " " << 1 << " ";
            else
                std::cout << " " << 0 << " ";
        }
        std::cout << std::endl;
    }

    for (int h = 0; h < hiddenSize; h++) {
        for (int o = 0; o < outputSize; o++) {
            hiddenStructure[h][o] = bestNet.getHiddenEdge(h, o);
            if (inputStructure[h][o])
                std::cout << " " << 1 << " ";
            else
                std::cout << " " << 0 << " ";
        }
        std::cout << std::endl;
    }
}

// TODO: How are heuristics being passed? Currently there are 2 2D vectors
// one for the input-hidden edges and one for the hidden-output edges
void Ants::createNetworkStructure(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                  std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    double inputDenom = getInputDenom();
    double hiddenDenom = getHiddenDenom();
                            //std::cout << "1" << std::endl;

    for (int a = 0; a < numAnts; a++) {
        //std::cout << "2" << std::endl;

        // initialize space for network TODO: check if actually auto-initializes to false/0?
        bool** tourFromInputLayer = new bool*[inputSize];
        bool** tourFromHiddenLayer = new bool*[hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            tourFromInputLayer[i] = new bool[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                tourFromInputLayer[i][j] = false;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            tourFromHiddenLayer[i] = new bool[outputSize];
            for (int j = 0; j < outputSize; j ++) {
                tourFromHiddenLayer[i][j] = false;
            }
        }

        
        std::pair<double, double> minMaxInput = getMinMaxInputProb();
        std::pair<double, double> minMaxHidden = getMinMaxHiddenProb();
        minMaxInput.first = minMaxInput.first / getInputDenom();
        minMaxInput.second = minMaxInput.second / getInputDenom();
        minMaxHidden.first = minMaxHidden.first / getHiddenDenom();
        minMaxHidden.second = minMaxHidden.second / getHiddenDenom();
        
        // std::cout << "MINMAX: " << minMaxInput.first << std::endl;
        // std::cout << "MINMAX: " << minMaxInput.second << std::endl;
        // std::cout << "MINMAX: " << minMaxHidden.first << std::endl
        // std::cout << "MINMAX: " << minMaxHidden.second << std::endl;

        for (int i = 0; i < inputSize; i++) {
            for (int h = 0; h < hiddenSize; h++) {
                double randNum = static_cast<double>(rand()) / RAND_MAX;
                double oldPInput = getProbabilityInput(i, h, inputDenom, inputHeuristics[i][h]);
                // std::cout << "OLD: " << oldPInput << std::endl;
                // std::cout << "ADJ: " << getAdjustedProbability(oldPInput, minMaxInput.first, minMaxInput.second) << std::endl;
                if (getAdjustedProbability(oldPInput, minMaxInput.first, minMaxInput.second) > randNum) {
                    tourFromInputLayer[i][h] = true;
                }
            }
        }

        for (int h = 0; h < hiddenSize; h++) {
            for (int o = 0; o < outputSize; o++) {
                double randNum = static_cast<double>(rand()) / RAND_MAX;
                double oldPHidden = getProbabilityHidden(h, o, hiddenDenom, hiddenHeuristics[h][o]);
                if (getAdjustedProbability(oldPHidden, minMaxHidden.first, minMaxHidden.second) > randNum) {
                    tourFromHiddenLayer[h][o] = true;
                }
            }
        }
                                    //std::cout << "5" << std::endl;

        // add new network to vector of networks for the iteration
        Net newNet(originalNet.getEpochs(), hiddenSize, originalNet.getLearningRate(), training, tourFromInputLayer, tourFromHiddenLayer);
                                            //std::cout << "5a" << std::endl;

        networks.push_back(newNet);
                                    //std::cout << "6" << std::endl;

    }

    // test all the networks and find the best one
    for (unsigned int i = 0; i < networks.size(); i++) {
        double currPercent = networks[i].reportErrorOnTestingSet(testing);
        if (currPercent > bestPercent) {
            bestPercent = currPercent;
            bestNet = networks[i];
        }
    }

    networks.clear();
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

double Ants::getAdjustedProbability(double p, double l, double h) {
    double s = 1 / (h - l);
    return s*(p - l);
}

// TODO: Check if actually returning a double
double Ants::getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic) {
    double prob = (pow(pheromoneFromInputLayer[inputNode][hiddenNode], alpha) * pow(heuristic, beta)) / denom;
    // std::cout << "H " << heuristic << "   Input prob: " << prob << std::endl;
    return prob;
}

// TODO heuristic
double Ants::getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic) {
    double prob = (pow(pheromoneFromHiddenLayer[hiddenNode][outputNode], alpha) * pow(heuristic, beta)) / denom;
    // std::cout << "H " << heuristic << "   Hidden prob: " << prob << std::endl;
    return prob;
}

// TODO heuristic
double Ants::getInputDenom() {
    double total = 0.0;
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            total += pow(pheromoneFromInputLayer[i][j], alpha) * pow(inputHeuristics[i][j], beta);
        }
    }
    return total;
}

std::pair<double, double> Ants::getMinMaxInputProb() {
    std::pair<double, double> minMax;
    minMax.first = std::numeric_limits<double>::max();
    minMax.second = std::numeric_limits<double>::min();
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            double p = pow(pheromoneFromInputLayer[i][j], alpha) * pow(inputHeuristics[i][j], beta);
            if (minMax.first > p) minMax.first = p;
            if (minMax.second < p) minMax.second = p;
        }
    }
    return minMax;
}

// TODO 2d heuristics to calculate actual value
double Ants::getHiddenDenom() {
    double total = 0.0;
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            total += pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(hiddenHeuristics[i][j], beta);
        }
    }
    return total;
}

std::pair<double, double> Ants::getMinMaxHiddenProb() {
    std::pair<double, double> minMax;
    minMax.first = std::numeric_limits<double>::max();
    minMax.second = std::numeric_limits<double>::min();
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            double p = pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(hiddenHeuristics[i][j], beta);
            if (minMax.first > p) minMax.first = p;
            if (minMax.second < p) minMax.second = p;
        }
    }
    return minMax;
}