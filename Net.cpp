#include "Net.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

Net::Net(int epochs_, 
         int hiddenLayerSize_,
         double learningRate_, 
         std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
         bool** inputStructure_,
         bool** hiddenStructure_)
    : epochs(epochs_), 
      inputSize(training[0].first.size() + 1),
      hiddenLayerSize(hiddenLayerSize_),
      outputSize(training[0].second.size()),
      learningRate(learningRate_),
      inputStructure(inputStructure_),
      hiddenStructure(hiddenStructure_),
      totalError(0.0)
{
    // Seed random number generator
    srand(time(NULL));
    
    // TODO change for connect 4?
    for (unsigned int i = 0; i < training.size(); i++) {
        training[i].first.push_back(1.0);
    }
    
    // Create weightsFromInputLayer
    weightsFromInputLayer = new double*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        weightsFromInputLayer[i] = new double[hiddenLayerSize];
    }

    // Initialize weightsFromInputLayer randomly between interval
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenLayerSize; j++) {
            // NOTE between -1 and 1
            weightsFromInputLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
        }
    }

    // Create weightsFromHiddenLayer
    weightsFromHiddenLayer = new double*[hiddenLayerSize];
    for (int i = 0; i < hiddenLayerSize; i++) {
        weightsFromHiddenLayer[i] = new double[outputSize];
    }

    // Initialize weightsFromHiddenLayer randomly between interval
    for (int i = 0; i < hiddenLayerSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // NOTE between -1 and 1
            weightsFromHiddenLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
        }
    }

    hiddenLayer = new Node[hiddenLayerSize];

    train(training, hiddenLayer);

}

void Net::evaluate(const std::vector<double>& input, std::vector<double>& output) const
{
    // For every hidden node    
    for (int i = 0; i < hiddenLayerSize; i++) {

        // Sum the weighting input
        double sum = 0;
        for (unsigned int j = 0; j < input.size(); j++) {
            if(inputStructure[j][i]) {
                sum += weightsFromInputLayer[j][i]*input[j];
            }
        }
        // And apply the sigmoid
        // std::cout << "Hidden Sum: " << sum << std::endl;
        hiddenLayer[i].evaluateNode(sum);

        // std::cout << hiddenLayer[i].getOutput() << " ";
    }

    // For every output node
    for (unsigned int i = 0; i < output.size(); i++) {

        // Sum the weighting input
        double sum = 0;
        for (int j = 0; j < hiddenLayerSize; j++) {
            if(hiddenStructure[j][i])
                sum += weightsFromHiddenLayer[j][i]*hiddenLayer[j].getOutput();
        }

        // And apply the sigmoid
        output[i] = Node::sigmoidActivation(sum);
        // std::cout << "Output Sum: " << sum << std::endl;
        // std::cout << "and ouput: " << output[i] << std::endl;
    }
}

void Net::train(std::vector<std::pair<std::vector<double>, std::vector<double>>>& data, 
                Node* hiddenLayer)
{
    // Loop over training set epochs times
    for (int i = 0; i < epochs; i++) {

        // Loop over training set
        for (unsigned int j = 0; j < data.size(); j++) {

            // Unpack example
            const std::vector<double>& exampleInput = data[j].first;
            const std::vector<double>& exampleOutput = data[j].second;

            std::vector<double> computedOutput(exampleOutput.size(), 0);

            // Evaluate network on example
            evaluate(exampleInput, computedOutput);

            totalError = 0.0;
            std::vector<double> error(computedOutput.size(), 0);

            // calculate the error at each output node
            for (unsigned int k = 0; k < computedOutput.size(); k++) {
                error[k] = exampleOutput[k] - computedOutput[k];
                totalError += fabs(error[k]);
                // std::cout << "Error " << k << "  " << error[k] << std::endl;
            }

            // now backprop backwards down the neural network, moving from ouput-->hidden to hidden-->input
            // weight adjustments

            // here, just do hidden-->ouput, but keep track of the weighted sum
            // for input-->hidden
            // TODO this only works if all the nodes are inter-connected
            std::vector<double> weightedErrorSum(hiddenLayerSize, 0);

            if (totalError != 0) {
                for (unsigned int k = 0; k < computedOutput.size(); k++) {
                    double gPOut = Node::sigmoidPrimeOutput(computedOutput[k]);
                    for (int l = 0; l < hiddenLayerSize; l++) {
                        if (hiddenStructure[l][k]) {
                            weightedErrorSum[l] += weightsFromHiddenLayer[l][k] * error[k] * gPOut;
                            weightsFromHiddenLayer[l][k] += learningRate * hiddenLayer[l].getOutput() * error[k] * gPOut;
                            // std::cout << "Error: " << error[k] << " Output: " << hiddenLayer[l].getOutput() << " gPOut: " << gPOut << std::endl;
                        }
                    }
                }

                for (int k = 0; k < hiddenLayerSize; k++) {
                    for (unsigned int l = 0; l < exampleInput.size(); l++) {
                        if (inputStructure[l][k])   
                            weightsFromInputLayer[l][k] += learningRate * exampleInput[l] * weightedErrorSum[k] * Node::sigmoidPrimeOutput(hiddenLayer[l].getOutput());
                        // std::cout << "LR: " << learningRate << " exampleInput: " << exampleInput[l] << " weighted sum: " << weightedErrorSum[k] << " prime: " << Node::sigmoidPrimeOutput(hiddenLayer[l].getOutput()) << std::endl;
                    }
                }
            }
        }

        std::cout << "Error " << totalError << " in epoch " << i << std::endl;
        // std::cout << "Input weights: " << std::endl;
        // for (int h = 5; h >= 0; h--) {
        //     for (int w = 0; w <= 6; w++) {
        //         std::cout << weightsFromInputLayer[w][h] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // std::cout << "Hidden weights: " << std::endl;
        // for (int i = 0; i < hiddenLayerSize; i++) {
        //     for (int k = 0; k < outputSize; k++) {
        //         std::cout << " " << weightsFromHiddenLayer[i][k] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
}

double Net::reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    int numCorrect = 0;
    totalError = 0.0;
    // Loop over testing set
    for (unsigned int i = 0; i < testing.size(); i++) {

        // Evaluate testing set
        std::vector<double> output(testing[i].second.size(), 0);
        evaluate(testing[i].first, output);

        double outMax = 0.0;
        int outNode = 0;

        for (unsigned int j = 0; j < testing[i].second.size(); j++) {
            totalError += testing[i].second[j] - output[j];
        }

        double labeledMax = 0.0;
        int labeledNode = 0;
        for (unsigned int j = 0; j < output.size(); j++) {
            if (output[j] > outMax) {
                outMax = output[j];
                outNode = j;
            }
            if (testing[i].second[j] > labeledMax) {
                labeledMax = testing[i].second[j];
                labeledNode = j;
            }
        }

        if (outNode == labeledNode) {
            numCorrect++;
        }
    }

    // Report to stdout
    double percentCorrect = static_cast<double>(numCorrect) / testing.size();
    std::cout << "Percentage correct on testing set: " << percentCorrect * 100 << "\%" << std::endl;
    return totalError;
}


