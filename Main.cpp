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

    // assemble fully connected network structures for the first network to work with
    // first input->hidden
    bool** fullInputStructure = new bool*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        fullInputStructure[i] = new bool[hiddenLayerSize];
        for (int j = 0; j < hiddenLayerSize; j++)
            fullInputStructure[i][j] = true;
    }

    // then hidden->output
    bool** fullHiddenStructure = new bool*[hiddenLayerSize];
    for (int i = 0; i < hiddenLayerSize; i++) {
        fullHiddenStructure[i] = new bool[outputSize];
        for (int j = 0; j < outputSize; j++)
            fullHiddenStructure[i][j] = true;
    }

    // Construct the fist network and print results
    std::cout << "Building and training first network" << std::endl;

	Net net(epochs, hiddenLayerSize, learningRate, training, fullInputStructure, fullHiddenStructure);
    net.reportErrorOnTestingSet(testing);

    std::cout << "Initial Network completed" << std::endl;

    bool** bestInputStructure = new bool*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        bestInputStructure[i] = new bool[hiddenLayerSize];
        for (int j = 0; j < hiddenLayerSize; j++)
            bestInputStructure[i][j] = false;
    }


    bool** bestHiddenStructure = new bool*[hiddenLayerSize];
    for (int i = 0; i < hiddenLayerSize; i++) {
        bestHiddenStructure[i] = new bool[outputSize];
        for (int j = 0; j < outputSize; j++)
            bestHiddenStructure[i][j] = false;
    }

    // then train/test on ants, man, forever
    if (atoi(argv[5]) == 1) {
        std::cout << "Beginning Ants" << std::endl;

        // setting some hard-coded parameters
        int numAnts = 5;
        double evaporationFactor = 0.1;
        double alpha = 1;
        double beta = 3;
        
        Ants ants(numAnts, evaporationFactor, alpha, beta, net);

        std::cout << "Ants Constructed" << std::endl;

        int numIterations = atoi(argv[6]);

        std::cout << "Ants Running" << std::endl;
        ants.run(numIterations, bestInputStructure, bestHiddenStructure, training, testing);

        std::cout << "Ants Completed" << std::endl << "Constructing best net from Ant data" << std::endl;

        // construct/train, then test the bet network
        Net antNet(epochs, hiddenLayerSize, learningRate, training, bestInputStructure, bestHiddenStructure);
        antNet.reportErrorOnTestingSet(testing);
    }

    // destruct the newed arrays
    std::cout << "Clearing some memory" << std::endl;

    for (int i = 0; i < inputSize; i++) {
        delete[] fullInputStructure[i];
        delete[] bestInputStructure[i];
    }
    delete[] fullInputStructure;
    delete[] bestInputStructure;

    for (int h = 0; h < hiddenLayerSize; h++) {
        delete[] fullHiddenStructure[h];
        delete[] bestHiddenStructure[h];
    }
    delete[] fullHiddenStructure;
    delete[] bestHiddenStructure;

    std::cout << "Done" << std::endl;

	return 0;
}
