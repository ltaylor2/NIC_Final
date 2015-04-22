#pragma once

#include <string>
#include <vector>

void readData32(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>& examples32);
void readData8(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>& examples8);
void printData(std::vector<std::pair<std::vector<double>, std::vector<double>>> examples, int width, int numExamples);
