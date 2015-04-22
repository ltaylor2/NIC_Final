#pragma once

#include <vector>

class Node {
public:
	Node();
	double evaluateNode(double input);
	double getOutput() const { return output; }

private:
    static double sigmoidActivation(double x);

	double output;
};
