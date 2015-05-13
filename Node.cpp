#include "Node.h"

#include <cmath>
#include <iostream>

Node::Node()
	: input(0),
	  output(0)
{}

double Node::evaluateNode(double input_) {
	input = input_;
	output = sigmoidActivation(input);
	return output;
}

double Node::sigmoidActivation(double x) {
    return 1/(1 + exp(-x));
}

double Node::sigmoidPrimeOutput(double output) {
    return output * (1 - output);
}
