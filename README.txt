A C++ Implementation of a Perceptron Network for Digit Recognition
Dan Cohen, Josh Imhoff, Liam Taylor

BUILD
    make

USAGE
    ./nn trainingFile testingFile inputRepresentation [0 = 32x32 | 1 = 8x8] epochs learningRate outputRepresentation [0 = single | 1 = multiple]

NOTES
    - will report error on training set and percentage correct on testing set for each epoch
