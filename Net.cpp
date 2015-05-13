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
    
    // Add bias node
    for (unsigned int i = 0; i < training.size(); i++) {
        training[i].first.push_back(1.0);
    }
    
    // Create weightsFromInputLayer
    weightsFromInputLayer = new double*[inputSize];
    inputHeuristics = new double*[inputSize];
    for (int i = 0; i < inputSize; i++) {
        weightsFromInputLayer[i] = new double[hiddenLayerSize];
        inputHeuristics[i] = new double[hiddenLayerSize];
    }

    // Initialize weightsFromInputLayer randomly between interval -1 and 1
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenLayerSize; j++) {
            weightsFromInputLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
            inputHeuristics[i][j] = 0;
        }
    }

    // Create weightsFromHiddenLayer
    weightsFromHiddenLayer = new double*[hiddenLayerSize];
    hiddenHeuristics = new double*[hiddenLayerSize];
    for (int i = 0; i < hiddenLayerSize; i++) {
        weightsFromHiddenLayer[i] = new double[outputSize];
        hiddenHeuristics[i] = new double[outputSize];
    }

    // Initialize weightsFromHiddenLayer randomly between interval -1 and 1
    for (int i = 0; i < hiddenLayerSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            weightsFromHiddenLayer[i][j] = 2*(static_cast<double>(rand()) / RAND_MAX) - 1;
            hiddenHeuristics[i][j] = 0;
        }
    }

    // Create hidden layer
    hiddenLayer = new Node[hiddenLayerSize];

    // Train NN
    train(training);

    // Remove bias nodes from training set afte training
    for (unsigned int i = 0; i < training.size(); i++)
        training[i].first.pop_back();

    // Finish updating heuristic information
    for (int i = 0; i < inputSize; i++) {
        for (int h = 0; h < hiddenLayerSize; h++) {
            inputHeuristics[i][h] = (double)inputHeuristics[i][h] / (epochs * training.size());
        }
    }
    for (int h = 0; h < hiddenLayerSize; h++) {
        for (int o = 0; o < outputSize; o++) {
            hiddenHeuristics[h][o] = (double)hiddenHeuristics[h][o] / (epochs * training.size());
        }
    }
}

Net::~Net()
{
    // IMPORANT memory leak, due to limitations in our design, getting rid of the leak
    //          would require SIGNIFICANT refactoring
    // NOTE the memory leak will not affect the running program

    // for (int i = 0; i < inputSize; i++) {
    //     delete[] weightsFromInputLayer[i];
    //     delete[] inputHeuristics[i];
    // }
    // delete[] weightsFromInputLayer;
    // delete[] inputHeuristics;

    // for (int i = 0; i < hiddenLayerSize; i++) {
    //     delete[] weightsFromHiddenLayer[i];
    //     delete[] hiddenHeuristics[i];
    // }
    // delete[] weightsFromHiddenLayer;
    // delete[] hiddenHeuristics;

    // delete[] hiddenLayer;
}

void Net::evaluate(const std::vector<double>& input, std::vector<double>& output) const
{
    // For every hidden node    
    for (int i = 0; i < hiddenLayerSize; i++) {
        // Sum the weighting input
        double sum = 0;
        for (unsigned int j = 0; j < input.size() - 1; j++) {
            if(inputStructure[j][i]) {
                sum += weightsFromInputLayer[j][i]*input[j];
            }
        }

        // And apply the sigmoid
        hiddenLayer[i].evaluateNode(sum);
    }

    // For every output node
    for (unsigned int i = 0; i < output.size(); i++) {
        // Sum the weighting input
        double sum = 0;
        for (int j = 0; j < hiddenLayerSize; j++) {
            if(hiddenStructure[j][i]) {
                sum += weightsFromHiddenLayer[j][i]*hiddenLayer[j].getOutput();
            }

        }

        // And apply the sigmoid
        output[i] = Node::sigmoidActivation(sum);
    }
}

void Net::train(std::vector<std::pair<std::vector<double>, std::vector<double>>>& data)
{
    // Loop over training set epochs times
    for (int i = 0; i < epochs; i++) {
        // Loop over training set
        for (unsigned int j = 0; j < data.size(); j++) {
            // Unpack example
            const std::vector<double>& exampleInput = data[j].first;
            const std::vector<double>& exampleOutput = data[j].second;

            // Evaluate network on example
            std::vector<double> computedOutput(exampleOutput.size(), 0);
            evaluate(exampleInput, computedOutput);

            // Calculate the error at each output node
            totalError = 0.0;
            std::vector<double> error(computedOutput.size(), 0);

            for (unsigned int k = 0; k < computedOutput.size(); k++) {
                error[k] = exampleOutput[k] - computedOutput[k];
                totalError += fabs(error[k]);
            }

            // Now backprop backwards down the neural network, moving from 
            // ouput-->hidden to hidden-->input weight adjustments

            // Here, just do hidden-->ouput, but keep track of the weighted sum
            // for input-->hidden
            std::vector<double> weightedErrorSum(hiddenLayerSize, 0);
            for (unsigned int k = 0; k < computedOutput.size(); k++) {
                double gPOut = Node::sigmoidPrimeOutput(computedOutput[k]);
                for (int l = 0; l < hiddenLayerSize; l++) {
                    if (hiddenStructure[l][k]) {   
                        weightedErrorSum[l] += weightsFromHiddenLayer[l][k] * error[k] * gPOut;
                        double weightChange = learningRate * hiddenLayer[l].getOutput() * error[k] * gPOut;
                        if (weightChange != 0)
                            hiddenHeuristics[l][k]++;
                        weightsFromHiddenLayer[l][k] += weightChange;
                    }
                }
            }
            // and now do hidden
            for (int k = 0; k < hiddenLayerSize; k++) {
                for (unsigned int l = 0; l < exampleInput.size(); l++) {
                    if (inputStructure[l][k]) {
                        double weightChange = learningRate * exampleInput[l] * weightedErrorSum[k] * Node::sigmoidPrimeOutput(hiddenLayer[k].getOutput());
                        if (weightChange != 0)
                            inputHeuristics[l][k]++;
                        weightsFromInputLayer[l][k] += weightChange;
                    }
                }
            }
        }
    }

    // Report error on stdout
    std::cout << "Error " << totalError << " in epoch " << i << std::endl;
}

double Net::reportErrorOnTestingSet(std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    int numCorrect = 0;
    totalError = 0.0;

    // for every testing sample data
    for (unsigned int i = 0; i < testing.size(); i++) {
        std::vector<double> output(testing[i].second.size(), 0);
        // evaluate the error on the set
        evaluate(testing[i].first, output);

        double outMax = 0.0;
        int outNode = 0;

        // add up the total error of the evaluation by adding up differences in every output node
        // to the actual labeled answer
        for (unsigned int j = 0; j < testing[i].second.size(); j++) {
            totalError += testing[i].second[j] - output[j];
        }

        double labeledMax = 0.0;
        int labeledNode = 0;
        // find the output node that represents the correct answer
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

        // and if it's correct, add one to the counter
        if (outNode == labeledNode) {
            numCorrect++;
        }
    }

    // the total percentage calculation for this network's testing
    double percentCorrect = static_cast<double>(numCorrect) / testing.size();
    std::cout << "Percentage correct on testing set: " << percentCorrect * 100 << "\%" << std::endl;
    return percentCorrect;
}
