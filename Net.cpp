#include "Net.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

Net::Net(int epochs, 
         int hiddenLayerSize_,
         double learningRate, 
         std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
         std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
    : hiddenLayerSize(hiddenLayerSize_)
{
    // Seed random number generator
    srand(time(NULL));
    
    // Add bias node
    for (unsigned int i = 0; i < training.size(); i++) {
        training[i].first.push_back(1.0);
    }
    
    // Input and output sizes
    int inputSize = training[0].first.size();
    int outputSize = training[0].second.size();

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
        weightsFromInputLayer[i] = new double[outputSize];
    }

    // Initialize weightsFromHiddenLayer randomly between interval
    for (int i = 0; i < hiddenLayerSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // NOTE between -1 and 1
            weightsFromHiddenLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
        }
    }

    // Create hiddenLayer
    hiddenLayer = new Node[hiddenLayerSize];

    train(epochs, learningRate, training, testing, hiddenLayer);
}

void Net::evaluate(const std::vector<double>& input, std::vector<double>& output) const
{
    // For every hidden node
    for (int i = 0; i < hiddenLayerSize; i++) {

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
        for (int j = 0; j < hiddenLayerSize; j++) {
            sum += weightsFromHiddenLayer[j][i]*hiddenLayer[j].getOutput();
        }

        // And apply the sigmoid
        output[i] = Node::sigmoidActivation(sum);
    }
}

void Net::train(int epochs, 
                double learningRate, 
                std::vector<std::pair<std::vector<double>, std::vector<double>>>& training, 
                std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing,
                Node* hiddenLayer)
{
    // Loop over training set epochs times
    for (int i = 0; i < epochs; i++) {

        // For reporting error on stdout
        double totalError = 0;

        // Loop over training set
        for (unsigned int j = 0; j < training.size(); j++) {

            // Unpack example
            const std::vector<double>& exampleInput = training[j].first;
            const std::vector<double>& exampleOutput = training[j].second;
            std::vector<double> computedOutput(exampleOutput.size(), 0);
            
            // Evaluate network on example
            evaluate(exampleInput, computedOutput);

            std::vector<double> error(computedOutput.size(), 0);

            // calculate the error at each node
            double totalError = 0;

            for (unsigned int k = 0; k < computedOutput.size(); k++) {
                error[k] = exampleOutput[k] - computedOutput[k];              
                totalError += fabs(error[k]);
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
                        weightedErrorSum[l] += weightsFromHiddenLayer[l][k] * error[k] * gPOut;
                        weightsFromHiddenLayer[l][k] += learningRate * hiddenLayer[l].getInput() * error[k] * gPOut;
                    }
                }

                for (int k = 0; k < hiddenLayerSize; k++) {
                    for (unsigned int l = 0; l < exampleInput.size(); l++) {
                        weightsFromInputLayer[l][k] += learningRate * exampleInput[l] * weightedErrorSum[k] * Node::sigmoidPrimeOutput(hiddenLayer[l].getOutput());
                    }
                }
            }
        }
    }
}

void Net::reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing) const
{
    int numCorrect = 0;

    // Loop over testing set
    for (unsigned int i = 0; i < testing.size(); i++) {

        // Evaluate testing set
        std::vector<double> output(testing[i].second.size(), 0);
        evaluate(testing[i].first, output);

        double outMax = 0.0;
        int outNode = 0;

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
}


