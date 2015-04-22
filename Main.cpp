#include "Perceptron.h"
#include "ReadFile.h"

#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    // Check cmd options
	if (argc != 7) {
		std::cout << "USAGE trainingFile testingFile inputType [0 = 32x32 | 1 = 8x8] epoch learningRate outputType[0 = single node | 1 = multiple nodes]" << std::endl;
		return 1;
	}

    // Read in data
	std::vector<std::pair<std::vector<double>, std::vector<double>>> training;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> testing;
    bool inputType = static_cast<bool>(atoi(argv[3]));
    if (inputType) {
        readData8(argv[2], training);
        readData8(argv[1], testing);
    }
    else {
        readData32(argv[2], training);
        readData32(argv[1], testing);
    }

    // Parse parameters
    int epochs = atoi(argv[4]);
    double learningRate = atof(argv[5]);
    bool multipleOutputs = static_cast<bool>(atoi(argv[6]));
    
    // Convert to multi-output format
    if (multipleOutputs) {
        for (unsigned int i = 0; i < training.size(); i++) {
            int recorded = static_cast<int>(training[i].second[0]);
            training[i].second = std::vector<double>(10, 0);
            training[i].second[recorded] = 1;
        }
        for (unsigned int i = 0; i < testing.size(); i++) {
            int recorded = static_cast<int>(testing[i].second[0]);
            testing[i].second = std::vector<double>(10, 0);
            testing[i].second[recorded] = 1;
        }
    }

    // Construct the network and print results
	Perceptron perceptron(epochs, learningRate, multipleOutputs, training, testing);

	return 0;
}
