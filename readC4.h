#pragma once

#include <vector>
#include <string>

// reading in Connect-Four data set
void readDataC4(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>&training,
                std::vector<std::pair<std::vector<double>, std::vector<double>>>&testing);

// output to view or double-check
void printData(std::vector<std::pair<std::vector<double>, std::vector<double>>> examples);
