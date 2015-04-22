#include "Net.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

Net::Net(int epochs, 
         int hiddenSize_,
         double learningRate, 
         std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
         std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
    : hiddenSize(hiddenSize_)
{
    // Seed random number generator
    srand(time(NULL));
    
    // Add bias node
    for (unsigned int i = 0; i < training.size(); i++) {
        training[i].first.push_back(1.0);
    }
    
    // Input and output sizes
    int inputSize = training[0].first.size();
    int outputSize = (multipleOutputs ? 10 : 1);

    // Create weightsFromInputLayer
    weightsFromInputLayer = new double*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        weightsFromInputLayer[i] = new double[hiddenSize];
    }

    // Initialize weightsFromInputLayer randomly between interval
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            // NOTE between -1 and 1
            weightsFromInputLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
        }
    }

    // Create weightsFromHiddenLayer
    weightsFromHiddenLayer = new double*[hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        weightsFromInputLayer[i] = new double[outputSize];
    }

    // Initialize weightsFromHiddenLayer randomly between interval
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // NOTE between -1 and 1
            weightsFromHiddenLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
        }
    }

    // Create hiddenLayer
    hiddenLayer = new Node[hiddenSize];

    // TODO train network
}

void Net::evaluate(const std::vector<double>& input, std::vector<double>& output) const
{
    // For every hidden node
    for (unsigned int i = 0; i < hiddenSize; i++) {

        // Sum the weighting input
        double sum = 0;
        for (unsigned int j = 0; j < input.size(); j++) {
            sum += weightsFromInputLayer[j][i]*input[j];
        }

        // And apply the sigmoid
        hiddenLayer[i].evaluateNode(sum);
    }

    // For every output node
    for (unsigned int i = 0; i < output.size(); i++) {

        // Sum the weighting input
        double sum = 0;
        for (unsigned int j = 0; j < hiddenSize; j++) {
            sum += weightsFromHiddenLayer[j][i]*hiddenLayer[j].getOutput();
        }

        // And apply the sigmoid
        output[i] = Node::sigmoidActivation(sum);
    }
}

// void Net::train(int epochs, 
//                 double learningRate, 
//                 bool multipleOutputs,
//                 std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
//                 std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
// {
//     // Loop over training set epochs times
//     for (int i = 0; i < epochs; i++) {
// 
//         // For reporting error on stdout
//         double totalError = 0;
// 
//         // Loop over training set
//         for (unsigned int j = 0; j < training.size(); j++) {
// 
//             // Unpack example
//             const std::vector<double>& exampleInput = training[j].first;
//             const std::vector<double>& exampleOutput = training[j].second;
//             std::vector<double> computedOutput(exampleOutput.size(), 0);
//             
//             // Evaluate network on example
//             evaluate(exampleInput, computedOutput);
// 
//             // Update weightsFromInputLayer for each node in output
//             for (unsigned int k = 0; k < computedOutput.size(); k++) {
// 
//                 // NOTE error calculated differently in different representations
//                 double error;
//                 if (multipleOutputs) {
//                     error = exampleOutput[k] - computedOutput[k];
//                 }
//                 else {
//                     error = exampleOutput[k] / 10 - computedOutput[k];
//                 }
//                 double sum = computedOutput[k];
//                 totalError += fabs(error);
//                 
//                 // If error is non-zero, update weightsFromInputLayer according to weight update rule 
//                 if (error != 0) {
//                     double gIn = sigmoidPrime(sum);
//                     for (unsigned int m = 0; m < exampleInput.size(); m++) {
//                         weightsFromInputLayer[m][k] += learningRate * error * exampleInput[m] * gIn;
//                     }
//                 }
//             }
//         }
// 
//         std::cout << "Epoch: " << i << ", error on training set: " << totalError << std::endl;
//         reportErrorOnTestingSet(multipleOutputs, testing);
//     }
// }
// 
// void Net::reportErrorOnTestingSet(bool multipleOutputs,
//                                   std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing) const
// {
//     int numCorrect = 0;
// 
//     // Loop over testing set
//     for (unsigned int i = 0; i < testing.size(); i++) {
// 
//         // Evaluate testing set
//         std::vector<double> output(testing[i].second.size(), 0);
//         evaluate(testing[i].first, output);
// 
//         // NOTE error calculated differently in different representations
//         if (multipleOutputs) {
//             double outMax = 0.0;
//             int outNode = 0;
// 
//             double labeledMax = 0.0;
//             int labeledNode = 0;
// 
//             for (unsigned int j = 0; j < output.size(); j++) {
//                 if (output[j] > outMax) {
//                     outMax = output[j];
//                     outNode = j;
//                 }
//                 if (testing[i].second[j] > labeledMax) {
//                     labeledMax = testing[i].second[j];
//                     labeledNode = j;
//                 }
//             }
//             if (outNode == labeledNode) {
//                 numCorrect++;
//             }
//         } else if (floor(10*output[0]) == testing[i].second[0]) {
//             numCorrect++;
//         }
//     }
// 
//     // Report to stdout
//     double percentCorrect = static_cast<double>(numCorrect) / testing.size();
//     std::cout << "Percentage correct on testing set: " << percentCorrect * 100 << "\%" << std::endl;
// }

double Net::sigmoidPrime(double sum) const {
    return exp(sum) / pow(exp(sum) + 1, 2);
}
