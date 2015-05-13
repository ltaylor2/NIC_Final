#pragma once

#include "Node.h"
#include "Net.h"

#include <queue>
#include <vector>

// Ants class designed specifically for construction of NN structure
class Ants {
public:
    // Constructor
    // @param numAnts_, number of ants creating NN structure
    // @param evaporationFactor_, as in standard ACO
    // @param beta_, as in standard ACO
    // @param net, neural network that ants are attempting to optimize structure for
    Ants(int numAnts_, double evaporationFactor_, double alpha_, double beta_, Net net);

    // Destructor
    ~Ants();

    // Run ACO/NN hyrbrid on training and testing data
    // @param numIterations, how many iterations of ACO
    // @param inputStructure, marks which edges are part of NN
    // @param hiddenStructure, marks which edges are part of NN
    // @param training, vector of labeled input and output used as training data
    // @param testing, vector of labeled input and output used as testing data
    void run(int numIterations,
             bool** inputStructure,
             bool** hiddenStructure,
             std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
             std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);

    // Helper method for creating NN with structure based on pheromone and heurstic 
    void createNetworkStructure(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing);

    // Helper methods for calculating/adjusting (on bell curve) probabilities of edge selection
    double getProbabilityInput(int inputNode, int hiddenNode, double denom, double heuristic);
    double getProbabilityHidden(int hiddenNode, int outputNode, double denom, double heuristic);
    double getInputDenom();  
    double getHiddenDenom();
    void updatePheromones(double error);

private:
    double getAdjustedProbability(double p, double l, double h);
    std::pair<double, double> getMeanStdevInputProb();
    std::pair<double, double> getMeanStdevHiddenProb();

    // Standard ACO params
    int numAnts;
    double evaporationFactor, alpha, beta;

    // NN params
    int inputSize, hiddenSize, outputSize;

    // Pheremone for each edge in fully connected NN
    double** pheromoneFromInputLayer;
    double** pheromoneFromHiddenLayer;

    // Stores "tours", particular NN structures selected by Ants
    bool** tourFromInputLayer;
    bool** tourFromHiddenLayer;
    bool** bestInputStructure;
    bool** bestHiddenStructure;

    // The original NN for which we are optimizing structure
    Net originalNet;

    // Heurstic is number of times weight in NN is changed during training
    double** inputHeuristics;
    double** hiddenHeuristics;
    
    // The optimized NN
    Net bestNet;
    double bestPercent;
    double heuristic;
   
    // Vector of NN having been found during ACO
    std::vector<Net> networks;
};

