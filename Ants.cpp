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

Ants::~Ants()
{
    for (int i = 0; i < inputSize; i++) {
        delete[] pheromoneFromInputLayer[i];
        delete[] tourFromInputLayer[i];
        delete[] bestInputStructure[i];
    }
    delete[] pheromoneFromInputLayer;
    delete[] tourFromInputLayer;
    delete[] bestInputStructure;
    
    for (int i = 0; i < hiddenSize; i++) {
        delete[] pheromoneFromHiddenLayer[i];
        delete[] tourFromHiddenLayer[i];
        delete[] bestHiddenStructure[i];
    }
    delete[] pheromoneFromHiddenLayer;
    delete[] tourFromHiddenLayer;
    delete[] bestHiddenStructure;
}

void Ants::run(int numIterations, bool** inputStructure, bool** hiddenStructure, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    for (int i = 0; i < numIterations; i++) {
        createNetworkStructure(training, testing);
        std::cout << "Updating pheromones" << std::endl;
        updatePheromones(bestNet.getTotalError());
        std::cout << "Completed Ant iteration " << i <<  std::endl;

    }

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

void Ants::createNetworkStructure(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                  std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    for (int a = 0; a < numAnts; a++) {
        // Initialize space for network
        // IMPORANT memory leak, due to limitations in our design, getting rid of the leak
        //          would require SIGNIFICANT refactoring
        // NOTE the memory leak will not affect the running program
        bool** tourFromInputLayer = new bool*[inputSize];
        for (int i = 0; i < inputSize; i++) {
            tourFromInputLayer[i] = new bool[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                tourFromInputLayer[i][j] = false;
            }
        }
        bool** tourFromHiddenLayer = new bool*[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            tourFromHiddenLayer[i] = new bool[outputSize];
            for (int j = 0; j < outputSize; j ++) {
                tourFromHiddenLayer[i][j] = false;
            }
        }
        
        double inputDenom = getInputDenom();
        double hiddenDenom = getHiddenDenom();

        // Get params for bell curve probability adjustment
        std::pair<double, double> meanStdevInput = getMeanStdevInputProb();
        std::pair<double, double> meanStdevHidden = getMeanStdevHiddenProb();

        // Create tours
        for (int i = 0; i < inputSize; i++) {
            for (int h = 0; h < hiddenSize; h++) {
                double randNum = 3.5 * static_cast<double>(rand()) / RAND_MAX - 0.5;
                double oldPInput = getProbabilityInput(i, h, inputDenom, inputHeuristics[i][h]);
                if (getAdjustedProbability(oldPInput, meanStdevInput.first, meanStdevInput.second) > randNum) {
                    tourFromInputLayer[i][h] = true;
                }
            }
        }
        for (int h = 0; h < hiddenSize; h++) {
            for (int o = 0; o < outputSize; o++) {
                double randNum = static_cast<double>(rand()) / RAND_MAX;
                double oldPHidden = getProbabilityHidden(h, o, hiddenDenom, hiddenHeuristics[h][o]);
                if (getAdjustedProbability(oldPHidden, meanStdevHidden.first, meanStdevHidden.second) > randNum) {
                    tourFromHiddenLayer[h][o] = true;
                }
            }
        }

        // Add new network to vector of networks for the iteration
        Net newNet(originalNet.getEpochs(), hiddenSize, originalNet.getLearningRate(), training, tourFromInputLayer, tourFromHiddenLayer);
        networks.push_back(newNet);
    }

    // Test all the networks and find the best one
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
    // Update pheromones according to standard ACO, minus evaporation
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

double Ants::getAdjustedProbability(double p, double mean, double stdev) {
    // Curves probabilities on bell curve, like in grading!
    return (p - mean) / stdev;
}

double Ants::getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic) {
    double prob = (pow(pheromoneFromInputLayer[inputNode][hiddenNode], alpha) * pow(heuristic, beta)) / denom;
    return prob;
}

double Ants::getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic) {
    double prob = (pow(pheromoneFromHiddenLayer[hiddenNode][outputNode], alpha) * pow(heuristic, beta)) / denom;
    return prob;
}

double Ants::getInputDenom() {
    double total = 0.0;
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            total += pow(pheromoneFromInputLayer[i][j], alpha) * pow(inputHeuristics[i][j], beta);
        }
    }
    return total;
}

std::pair<double, double> Ants::getMeanStdevInputProb() {
    std::pair<double, double> meanStdev;
    meanStdev.first = 0;
    meanStdev.second = 0;

    double denom = getInputDenom();
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            double score = pow(pheromoneFromInputLayer[i][j], alpha) * pow(inputHeuristics[i][j], beta) / denom;
            meanStdev.first += score;
        }
    }
    meanStdev.first = meanStdev.first / (inputSize*hiddenSize);

    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            double score = pow(meanStdev.first - pow(pheromoneFromInputLayer[i][j], alpha) * pow(inputHeuristics[i][j], beta) / denom, 2);
            meanStdev.second += score;

        }
    }
    meanStdev.second = meanStdev.second / (inputSize*hiddenSize);
    meanStdev.second = sqrt(meanStdev.second);
    return meanStdev;
}

double Ants::getHiddenDenom() {
    double total = 0.0;
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            total += pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(hiddenHeuristics[i][j], beta);
        }
    }
    return total;
}

std::pair<double, double> Ants::getMeanStdevHiddenProb() {
    std::pair<double, double> meanStdev;
    meanStdev.first = 0;
    meanStdev.second = 0;

    double denom = getHiddenDenom();
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            double score = pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(hiddenHeuristics[i][j], beta) / denom;
            meanStdev.first += score;
        }
    }
    meanStdev.first = meanStdev.first / (inputSize*hiddenSize);

    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            double score = pow(meanStdev.first - pow(pheromoneFromHiddenLayer[i][j], alpha) * pow(hiddenHeuristics[i][j], beta) / denom, 2);
            meanStdev.second += score;

        }
    }
    meanStdev.second = meanStdev.second / (inputSize*hiddenSize);
    meanStdev.second = sqrt(meanStdev.second);
    return meanStdev;
}
