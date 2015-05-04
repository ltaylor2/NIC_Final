#include "Net.h"
#include "readC4.h"

#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    // Check cmd options
	if (argc != 7) {
		std::cout << "USAGE dataFile epochs learningRate hiddenLayerSize ants[ 0 = false | 1 = true] antIterations" << std::endl;
		return 1;
	}

    // Read in data
	std::vector<std::pair<std::vector<double>, std::vector<double>>> training;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> testing;
    readDataC4(argv[1], training, testing);

    // Parse parameters
    int epochs = atoi(argv[2]);
    double learningRate = atof(argv[3]);

    int inputSize = training[0].first.size();
    int hiddenLayerSize = atoi(argv[4]);
    int outputSize = training[0].second.size();
    
    // Construct the fist network and print results
    bool** fullInputStructure = new bool[inputSize];
    for (int i = 0; i < inputSize; i++) {
        fullInputStructure[i] = new bool[hiddenLayerSize];
        fullInputStructure[i] = { true };
    }

    bool** fullHiddenStructure = new bool[hiddenLayerSize];
    for (int i = 0; i < outputSize; i++) {
        fullHiddenStructure[i] = new bool[outputSize];
        fullHiddenStructure[i] = { false };
    }

	Net net(epochs, hiddenLayerSize, learningRate, training, testing, fullInputStructure, fullHiddenStructure);
    net.reportErrorOnTestingSet(testing);

    // then train/test on ants, man, forever
    if (atoi(argv[5]) == 1) {
        int numAnts = 10;
        double evaporationFactor = 0.1
        double alpha = 1;
        double beta = 3;
        
        Ants ants(numAnts, evaporationFactor, alpha, beta, net);
        bool** inputStructure = new bool[inputSize];
        for (int i = 0; i < inputSize; i++) {
            inputStructure[i] = new bool[hiddenSize];
            inputStructure[i] = { false };
        }

        bool** hiddenStructure = new bool[hiddenSize];
        for (int i = 0; i < outputSize; i++) {
            hiddenStructure[i] = new bool[outputSize];
            hiddenStructure[i] = { false };
        }

        int numIterations = atoi(argv[6]);
        ants.run(numIterations, inputStructure, hiddenStructure, training, testing);
    }

    // train/test the resulting net
    Net antNet(epochs, hiddenLayerSize, learningRate, training, testing, inputStructure, hiddenStructure);
    antNet.reportErrorOnTestingSet(testing);

	return 0;
}
