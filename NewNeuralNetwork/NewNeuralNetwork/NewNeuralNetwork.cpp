// NewNeuralNetwork.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "Network.h"
#include "mnist.h"
#include "DQNAgent.h"
#include "EnvironmentMonster.h"

int testMnist()
{
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

void testDQN() {
	float learningRate = 0.005;
	float gamma = 0.99;
	int inputSize = 25;
	float epsilon = 1;
	int actionNumber = 4;
	int memorySize = 1000000;
	int batchSize = 64;
	float epsilonEnd = 0.01;

	int iteration = 0;

	DQNAgent agent(learningRate, gamma, actionNumber, epsilon, batchSize, inputSize, epsilonEnd, memorySize);

	Point startPos;
	startPos.x = 0;
	startPos.y = 0;
	EnvironmentMonster env(startPos);

	int gameNumber = 1000;

	vector<float> observation;
	vector<float> newObservation;

	for (int i = 0; i < gameNumber; i++) {
		bool done = false;
		float score = 0;
		observation = env.Reset();

		while (!done) {
			if (iteration % 50 == 0) {
				agent.UpdateTarget();
				cout << iteration << endl;
			}
			Action action = (Action)agent.ChooseAction(observation);
			newObservation = env.Step(action);
			float reward = env.getReward();
			done = env.getDone();
			score = env.getScore();

			agent.Remember(observation, action, reward, newObservation, done);
			observation = newObservation;
			agent.Learn();
			iteration++;
		}

		cout << "Game" << i << " Score: " << score << endl;
	}

	//Test
	vector<float> testIn(inputSize);
	for (int i = 0; i < inputSize; i++) {
		testIn[i] = 1;
		agent.ChooseActionExploit(testIn);
		testIn[i] = 0;
	}
	bool done = false;
	float score = 0;
	observation = env.Reset();

	while (!done) {
		env.PrintObservation();
		Action action = (Action)agent.ChooseActionExploit(observation);
		cout << "----------------------" << endl;
		newObservation = env.Step(action);
		float reward = env.getReward();
		done = env.getDone();
		score = env.getScore();

		observation = newObservation;
		iteration++;
	}
	env.PrintObservation();
}

int testNetwork() {
	int inputSize = 9;
	int outputSize = 4;
	int hiddenSize = 10;
	int hiddenNumber = 2;
	float learningRate = 0.01;
	Network net(inputSize, hiddenSize, hiddenNumber, outputSize, learningRate);

	int maxIterations = 50000;
	int batchSize = 9;

	vector<float> testIn(batchSize);
	vector<float> testOut(outputSize);

	vector<vector<float>> outputExamples(batchSize);
	outputExamples[0] = { 100, -1, -1, 100 };
	outputExamples[1] = { 100, -1, -1, 100 };
	outputExamples[2] = { -1, -1, -1, 100 };

	outputExamples[3] = { 100, -1, -1, 100 };
	outputExamples[4] = { 100, -1, -1, 100 };
	outputExamples[5] = { -1, -1, -1, 100 };

	outputExamples[6] = { 100, -1, -1, -1 };
	outputExamples[7] = { 100, -1, -1, -1 };
	outputExamples[8] = { 100, -1, -1, 100 };

	for (int k = 0; k < maxIterations; k++) {
		for (int i = 0; i < batchSize; i++) {
			testIn[i] = 1;
			net.Fit(testIn, outputExamples[i]);
			testIn[i] = 0;
		}
		if (k % 1000 == 0) {
			cout << "Epoch" << k << " Loss: " << net.resetLoss() / batchSize << endl;
		}
		else
			net.resetLoss();
		net.UpdateWeights();
	}

	//Test
	for (int i = 0; i < batchSize; i++) {
		testIn[i] = 1;
		testOut = net.Propagate(testIn);
		for (int j = 0; j < outputSize; j++)
			cout << testOut[j] << ", ";
		testIn[i] = 0;
		cout << endl;
	}
}

int main() {
	srand(time(NULL));

	testDQN();
	//testNetwork();
	//testMnist();

	system("pause");
}