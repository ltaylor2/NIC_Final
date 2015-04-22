#include "Node.h"

#include <cmath>

Node::Node()
	: output(0)
{}

double Node::evaluateNode(double input) {
	output = Node::sigmoidActivation(input);
	return output;
}

static double Node::sigmoidActivation(double x) const {
    return 1/(1+ exp(-x+0.5));
}
