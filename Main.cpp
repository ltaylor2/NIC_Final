#include "Net.h"
#include "Ants.h"
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

    int inputSize = training[0].first.size() + 1;
    int hiddenLayerSize = atoi(argv[4]);
    int outputSize = training[0].second.size();

    // Construct the fist network and print results
    bool** fullInputStructure = new bool*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        fullInputStructure[i] = new bool[hiddenLayerSize];
        for (int j = 0; j < hiddenLayerSize; j++)
            fullInputStructure[i][j] = true;
    }


    bool** fullHiddenStructure = new bool*[hiddenLayerSize];
    for (int i = 0; i < hiddenLayerSize; i++) {
        fullHiddenStructure[i] = new bool[outputSize];
        for (int j = 0; j < outputSize; j++)
            fullHiddenStructure[i][j] = true;
    }

	Net net(epochs, hiddenLayerSize, learningRate, training, fullInputStructure, fullHiddenStructure);
    net.reportErrorOnTestingSet(testing);

    bool** bestInputStructure = new bool*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        bestInputStructure[i] = new bool[hiddenLayerSize];
        for (int j = 0; j < hiddenLayerSize; j++)
            bestInputStructure[i][j] = false;
    }


    bool** bestHiddenStructure = new bool*[hiddenLayerSize];
    for (int i = 0; i < outputSize; i++) {
        bestHiddenStructure[i] = new bool[outputSize];
        for (int j = 0; j < outputSize; j++)
            bestHiddenStructure[i][j] = false;
    }

    // then train/test on ants, man, forever
    if (atoi(argv[5]) == 1) {
        int numAnts = 10;
        double evaporationFactor = 0.1;
        double alpha = 1;
        double beta = 3;
        
        std::cout << "11111" << std::endl;
        Ants ants(numAnts, evaporationFactor, alpha, beta, net);

        std::cout << "222222" << std::endl;

        int numIterations = atoi(argv[6]);
        ants.run(numIterations, bestInputStructure, bestHiddenStructure, training, testing);

        std::cout << "33333" << std::endl;

        Net antNet(epochs, hiddenLayerSize, learningRate, training, bestInputStructure, bestHiddenStructure);
        antNet.reportErrorOnTestingSet(testing);

        std::cout << "44444" << std::endl;

    }

	return 0;
}
