#pragma once

#include <vector>

// Node class used for hidden layer
class Node {
public:
    // Contructor
	Node();

    // Evaluate some (already summed) input_
	double evaluateNode(double input_);

    // Getters
	double getInput() const { return input; }
	double getOutput() const { return output; }

    // Helper methods for doing activation
    // NOTE public so that Net can use directly
	static double sigmoidActivation(double x);
	static double sigmoidPrimeOutput(double output);

private:
    double input;
	double output;
};
