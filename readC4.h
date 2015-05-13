#pragma once

#include <vector>
#include <string>

void readDataC4(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>&training,
                std::vector<std::pair<std::vector<double>, std::vector<double>>>&testing);
void printData(std::vector<std::pair<std::vector<double>, std::vector<double>>> examples);
