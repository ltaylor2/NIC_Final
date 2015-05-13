#include "readC4.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

void readDataC4(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
    std::fstream file(fp);
    if (file.is_open()) {
        std::string line;
        int lineCount = 0;
        while (getline(file, line)) {
            lineCount++;
            std::stringstream ss(line);
            std::string c;
            std::vector<double> exampleData;
            while (getline(ss, c, ',')) {
                if (c.compare("b") == 0) {
                    exampleData.push_back(0.0);
                }
                else if (c.compare("x") == 0) {
                    exampleData.push_back(1.0);
                }
                else if (c.compare("o") == 0) {
                    exampleData.push_back(-1.0);
                }
                else {
                    std::vector<double> resultData(3, 0);
                    if (c.compare("win") == 0) {
                        resultData[1] = 1.0;
                    }
                    else if (c.compare("loss") == 0) {
                        resultData[2] = 1.0;
                    }
                    else if (c.compare("draw") == 0) {
                        resultData[0] = 1.0;
                    }
                    else {
                        std::cout << "Something has gone terribly wrong." << std::endl;
                        break;
                    }
                    std::pair<std::vector<double>, std::vector<double>> example(exampleData, resultData);
                    if (lineCount < 30000)
                        training.push_back(example);
                    else
                        testing.push_back(example);
                }
            }
        }
    }
    else {
        std::cout << "File did not open properly." << std::endl;
    }
}

void printData(std::vector<std::pair<std::vector<double>, std::vector<double>>> examples) {
    for (unsigned int i = 0; i < examples.size(); i++) {
        for (unsigned int j = 0; j < examples[i].first.size(); j++) {
            std::cout << examples[i].first[j];
        }
        std::cout << "     ";
        for (unsigned int j = 0; j < examples[i].second.size(); j++) {
            std::cout << examples[i].second[j];
        }
        std::cout << std::endl;
    }
}
