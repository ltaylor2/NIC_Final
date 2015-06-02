#include "readBC.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

void readDataBC(std::string fp, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training,
                                std::vector<std::pair<std::vector<double>, std::vector<double>>>& testing)
{
	std::fstream file(fp);
	if (file.is_open()) {
		std::string line;
		int lineCount = 0;
		
		while (getline(file, line)) {
			if (line.size() == 0)
				break;
			lineCount++;
			std::stringstream ss(line);
			std::string c;
			std::vector<double> exampleData;

			// get the first line and throw it out
			getline(ss, c, ',');
			
			// now get Malignant or Benign and stuff to output
			getline(ss, c, ',');
			std::vector<double> resultData(2,0);
			if (c.compare("M") == 0)
				resultData[0] = 1.0;
			else if (c.compare("B") == 0)
				resultData[1] = 1.0;
			else {
				std::cout << "Something has gone terribly wrong" << std::endl;
				std::cout << lineCount << std::endl;
				break;
			}

			while (getline(ss, c, ',')) {
				exampleData.push_back(stod(c));
			}

			std::pair<std::vector<double>, std::vector<double>> example(exampleData, resultData);

            double randNum = static_cast<double>(rand()) / RAND_MAX;
			if (randNum < 0.30)
				training.push_back(example);
			else 
				testing.push_back(example);
		}
	}

	else
		std::cout << "File IO error" << std::endl;
}
