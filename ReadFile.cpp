#include "ReadFile.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

//Read 32x32 training/testing examples from file
void readData32(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>& examples32)
{
	std::fstream file(fp);
	std::stringstream ss;
	if (file.is_open()) {
		std::string line;
		std::vector<double> exampleData;
		while (getline(file, line)) {
			if (line[0] == '0' || line[0] == '1') {
				for (unsigned int i = 0; i < line.length(); i++) {
					double bit;
					ss.clear();
					ss.str("");
					ss << line[i];
					ss >> bit;
					exampleData.push_back(bit);
				}
			}
			else if (line.length() == 2){
				double digit;
				ss.clear();
				ss.str("");
				ss << line[1];
				ss >> digit;
				std::vector<double> label;
				label.push_back(digit);
				std::pair<std::vector<double>, std::vector<double>> example(exampleData, label);
				examples32.push_back(example);
				exampleData.clear();
			}
		}
	}
	else {
		std::cout << "File did not open properly." << std::endl;
	}
}

//Read 8x8 training/testing examples from file
void readData8(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>& examples8)
{
	std::fstream file(fp);
	std::stringstream ss;
	if (file.is_open()) {
		std::string line;
		std::vector<double> exampleData;
		while (getline(file, line)) {
			ss.clear();
			ss.str("");
			ss << line;
			double elem;
			while (ss >> elem) {
				exampleData.push_back(elem);
				if (ss.peek() == ',') {
					ss.ignore();
				}
			}
			int digit = exampleData.back();
			exampleData.pop_back();
			std::vector<double> label;
			label.push_back(digit);
			std::pair<std::vector<double>, std::vector<double>> example(exampleData, label);
			examples8.push_back(example);
			exampleData.clear();
		}
	}
	else {
		std::cout << "File did not open properly." << std::endl;
	}
}

//Prints data that has been read from file
void printData(std::vector<std::pair<std::vector<double>, std::vector<double>>> examples, int width, int numExamples) {
	for (int k = 0; k < numExamples; k++) {	
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < width; j++) {
				std::cout << examples[k].first[j+(width * i)];
			}
			std::cout << std::endl;
		}
		std::cout << examples[k].second[0] << std::endl;
	}
}

