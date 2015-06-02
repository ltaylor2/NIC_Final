#pragma once

#include <vector>
#include <string>

// read in WDBC data
void readDataBC(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>&training,
                std::vector<std::pair<std::vector<double>, std::vector<double>>>&testing);
