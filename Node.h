#pragma once

#include <vector>

class Node {
public:
	Node();
	double evaluateNode(double input_);
	double getOutput() const { return output; }
	double getInput() const { return input; }
	static double sigmoidActivation(double x);
	static double sigmoidPrimeOutput(double output);

private:
    double input;
	double output;
};
