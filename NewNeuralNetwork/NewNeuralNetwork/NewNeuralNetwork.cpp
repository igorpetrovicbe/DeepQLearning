// NewNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Network.h"
#include "mnist.h"

int main()
{
	srand(time(NULL));

	int inputSize = 784;
	int outputSize = 10;
	int hiddenSize = 50;
	int hiddenNumber = 2;
	float learningRate = 0.02;
	Network net(inputSize, hiddenSize, hiddenNumber, outputSize, learningRate);

	omp_set_num_threads(1);

	int maxIterations = 500000;
	int batchSize = 12;

	int datasetSize;

	vector<vector<float>>datasetInputs = read_mnist_images("train-images.idx3-ubyte", datasetSize, inputSize);
	vector<vector<float>>datasetOutputs = read_mnist_labels("train-labels.idx1-ubyte", datasetSize, outputSize);
	
	int validationSetSize = 10000;
	int trainingSetSize = datasetSize - validationSetSize;

	//Train
	int epoch = 1;
	double startTime = omp_get_wtime();
	double endTime;
	for (int i = 0; i < maxIterations; i++) {
		net.Fit(datasetInputs[i % trainingSetSize], datasetOutputs[i % trainingSetSize]);
		if (i % batchSize == 0 && i != 0)
			net.UpdateWeights();

		if (i % 1000 == 0) {
			endTime = omp_get_wtime();
			cout << i << "\tTook: " << (endTime - startTime) * 1000 << "ms." << endl;
			startTime = omp_get_wtime();
		}
		if (i % trainingSetSize == trainingSetSize - 1) {
			cout << "Epoch" << epoch << " Loss: " << net.resetLoss() / trainingSetSize << endl;
			epoch++;
			net.setLearningRate(learningRate / sqrt(epoch));
		}
	}
	endTime = omp_get_wtime();
	cout << endl << "\tTest finished. Took: " << (endTime - startTime) * 1000 << "ms." << endl << endl;

	//Calculate Accurracy
	vector<float> output(outputSize);
	int correct = 0;
	for (int k = trainingSetSize; k < datasetSize; k++) {
		output = net.Propagate(datasetInputs[k]);
		int choice = 0;
		for (int i = 1; i < outputSize; i++)
			if (output[i] > output[choice])
				choice = i;
		if (datasetOutputs[k][choice] == 1)
			correct++;
	}
	cout << "Accurracy: " << 100 * ((float)correct) / validationSetSize << "%" << endl;
	std::system("pause");

	//Calculate Accurracy on training set (for comparison)
	correct = 0;
	for (int k = 0; k < validationSetSize; k++) {
		output = net.Propagate(datasetInputs[k]);
		int choice = 0;
		for (int i = 1; i < outputSize; i++)
			if (output[i] > output[choice])
				choice = i;
		if (datasetOutputs[k][choice] == 1)
			correct++;
	}
	cout << "Accurracy on TS: " << 100 * correct / validationSetSize << "%" << endl;
	std::system("pause");

	//Print Test
	int testSize = 10;
	for (int i = 0; i < testSize; i++) {
		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++) {
				if (datasetInputs[i][j * 28 + k] > 0.5)
					cout << '#';
				else
					cout << ' ';
			}
			cout << endl;
		}
		output = net.Propagate(datasetInputs[i]);
		cout << '(';
		for (int j = 0; j < outputSize; j++) {
			cout << output[j] << ", ";
		}
		cout << ')' << endl << endl;
	}
	
	std::system("pause");

    return 0;
}

