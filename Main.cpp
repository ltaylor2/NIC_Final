#include "Net.h"

#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    // Check cmd options
	if (argc != 5) {
		std::cout << "USAGE dataFile epochs learningRate hiddenLayerSize" << std::endl;
		return 1;
	}

    // Read in data
	std::vector<std::pair<std::vector<double>, std::vector<double>>> training;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> testing;
    // readDataC4(argv[1], training, testing);

    // Parse parameters
    int epochs = atoi(argv[2]);
    double learningRate = atof(argv[3]);
    int hiddenLayerSize = atoi(argv[4]);
    
    // Construct the network and print results
	Net net(epochs, learningRate, hiddenLayerSize, training, testing);

	return 0;
}
