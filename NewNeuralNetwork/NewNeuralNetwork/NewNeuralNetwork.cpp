// NewNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Network.h"


int main()
{
	int inputSize = 2;
	int outputSize = 1;
	float learningRate = 0.1;
	Network net(inputSize, 10, 2, outputSize, learningRate);

	int maxIterations = 1000000;
	int batchSize = 4;

	int datasetSize = 4;

	vector<vector<float>> datasetInputs(datasetSize);
	datasetInputs[0] = { 0, 0 };
	datasetInputs[1] = { 0, 1 };
	datasetInputs[2] = { 1, 0 };
	datasetInputs[3] = { 1, 1 };

	vector<vector<float>> datasetOutputs(datasetSize);
	datasetOutputs[0] = { 0 };
	datasetOutputs[1] = { 1 };
	datasetOutputs[2] = { 1 };
	datasetOutputs[3] = { 0 };

	//Train
	for (int i = 0; i < maxIterations; i++) {
		net.Fit(datasetInputs[i % datasetSize], datasetOutputs[i % datasetSize]);
		if (i % batchSize == 0 && i != 0)
			net.UpdateWeights();

		if (i % 1000 == 0) {
			cout << i << endl;
		}
	}

	//Test
	vector<float> output(outputSize);
	for (int i = 0; i < datasetSize; i++) {
		output = net.Propagate(datasetInputs[i]);
		for (int j = 0; j < outputSize; j++) {
			cout << output[j] << ", ";
		}
		cout << endl;
	}

	system("pause");

    return 0;
}

