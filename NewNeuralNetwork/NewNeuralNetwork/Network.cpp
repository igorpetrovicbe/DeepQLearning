#include "stdafx.h"
#include "Network.h"

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float sigmoidDerivative(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

float relu(float x) { //leaky RELU
	if (x > 0)
		return x;
	return 0.01 * x;
}

float reluDerivative(float x) {
	if (x > 0)
		return 1;
	return 0.01;
}

float crossEntropy(vector<float> expected, vector<float> predicted) {
	float sum = 0;
	for (int i = 0; i < expected.size(); i++) {
		sum += expected[i] * log(predicted[i]);
	}
	return -sum;
}

float crossEntropyLoss(float predicted) {
	return -log(predicted);
}

float crossEntropyLossDerivative(float predicted) {
	return -1 / predicted;
}

float softmaxDerivative(float out1, float out2, bool same) {
	if (same)
		return out1 * (1 - out1);
	else
		return -out1 * out2;
}

void softmax(vector<float>& x) {
	float expSum = 0;
	float maxOut = x[0];
	for (int i = 0; i < x.size(); i++) {
		if (x[i] > maxOut)
			maxOut = x[i];
	}
	for (int i = 0; i < x.size(); i++)
		expSum += exp(x[i] - maxOut);
	for (int i = 0; i < x.size(); i++) {
		float shiftx = x[i] - maxOut;
		x[i] = exp(shiftx) / expSum;
	}
}

float activation(float x, Activation activation) {
	switch (activation) {
	case SIGMOID:
		return sigmoid(x);
		break;
	case RELU:
		return relu(x);
		break;
	case NONE:
		return x;
		break;
	default:
		return sigmoid(x);
		break;
	}
}

float activationDerivative(float x, Activation activation) {
	switch (activation) {
	case SIGMOID:
		return sigmoidDerivative(x);
		break;
	case RELU:
		return reluDerivative(x);
		break;
	case NONE:
		return 1;
		break;
	default:
		return sigmoidDerivative(x);
		break;
	}
}

float randomDeviation(float standardDeviation) {
	float r = rand() % 10000;
	r = (r - 5000) / 5000;
	return standardDeviation * r;
}

float gaussian(float standardDeviation) {
	float sampleSize = 100;
	float sum = 0;
	for (int i = 0; i < sampleSize; i++) {
		float r = rand() % 10000;
		r = (r - 5000) / 5000;
		sum += r;
	}
	return standardDeviation * sum / sampleSize;
}

float heInitialization(float number) {
	return gaussian(sqrt(2 / number));
}

float xavier(float number) {
	//float standardDeviation = sqrt(6) / sqrt(number);
	return randomDeviation(1);
}

float initWeight(float leftSize, float rightSize, Activation activation) {
	switch (activation)
	{
	case SIGMOID:
		return xavier(leftSize + rightSize);
		break;
	case RELU:
		return heInitialization(leftSize);
		break;
	case SOFTMAX:
		return xavier(leftSize + rightSize);
		break;
	case NONE:
		return heInitialization(leftSize);
		break;
	default:
		return xavier(leftSize + rightSize);
		break;
	}
}

void testMP() {
	printf("\nfirst case : no threading! \n ");
	double pi = 0.0;
	const int iterationCount = 200000000;
	clock_t startTime = clock();
	for (int i = 0; i < iterationCount; i++)
	{
		pi += 4 * (i % 2 ? -1 : 1) / (2.0 * i + 1.0);
	}
	printf("Elpase Time : %.3lf sec \n", (clock() - startTime) / 1000.);
	printf("pi = %.8f\n", pi);
	/////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	printf("\nsecond case : OpenMP - threading! \n ");
	pi = 0.0;
	startTime = clock();

#pragma omp parallel for
	for (int i = 0; i < iterationCount; i++)
	{
		pi += 4 * (i % 2 ? -1 : 1) / (2.0 * i + 1.0);
	}
	printf("Elpase Time : %.3lf sec \n", (clock() - startTime) / 1000.);
	printf("pi = %.8f\n", pi);
	/////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////
	printf("\nthird case : OpenMP - reduction set\n");
	pi = 0.0;
	startTime = clock();
}

Network::Network(int inputSize, int hiddenSize, int hiddenNumber, int outputSize, float learningRate) {
	this->inputSize = inputSize;
	this->hiddenSize = hiddenSize;
	this->hiddenNumber = hiddenNumber;
	this->outputSize = outputSize;
	this->learningRate = learningRate;
	batchCount = 0;

	//Init weights and biases
	float r;
	for (int i = 0; i < hiddenNumber; i++) {
		float previousSize, nextSize;
		if (i == 0)
			previousSize = inputSize;//+1 for bias?
		else
			previousSize = hiddenSize;

		if (i == hiddenNumber - 1)
			nextSize = outputSize;
		else
			nextSize = hiddenSize;

		vector<Neuron> newHidden(hiddenSize);
		for (int j = 0; j < hiddenSize; j++) {
			Neuron neuron;
			//neuron.bias = xavier(previousSize + nextSize);
			neuron.bias = initWeight(previousSize, nextSize, hiddenActivation);
			neuron.out = 0;
			neuron.gradient = 0;
			neuron.biasDelta = 0;
			neuron.momentum = 0;
			neuron.variance = 0;
			newHidden[j] = neuron;
		}
		hidden.push_back(newHidden);
	}
	float previousSize = hiddenSize;
	for (int i = 0; i < outputSize; i++) {
		Neuron neuron;
		neuron.bias = initWeight(previousSize, 0, outputActivation);
		neuron.out = 0;
		neuron.gradient = 0;
		neuron.biasDelta = 0;
		neuron.momentum = 0;
		neuron.variance = 0;
		output.push_back(neuron);
	}
	for (int k = 0; k <= hiddenNumber; k++) {
		int leftSize, rightSize;
		if (k == 0) {
			leftSize = inputSize;
			rightSize = hiddenSize;
		}
		else if (k == hiddenNumber) {
			leftSize = hiddenSize;
			rightSize = outputSize;
		}
		else {
			leftSize = hiddenSize;
			rightSize = hiddenSize;
		}
		vector<vector<float>> newLayerWeights(leftSize);
		vector<vector<float>> newLayerWeightsDelta(leftSize);
		vector<vector<float>> newLayerMomentums(leftSize);
		vector<vector<float>> newLayerVariances(leftSize);
		for (int i = 0; i < leftSize; i++) {
			vector<float> newWeights(rightSize);
			vector<float> newWeightsDelta(rightSize);
			vector<float> newMomentums(rightSize);
			vector<float> newVariances(rightSize);
			for (int j = 0; j < rightSize; j++) {
				if (k < hiddenNumber)
					newWeights[j] = initWeight(leftSize, rightSize, hiddenActivation);
				else
					newWeights[j] = initWeight(leftSize, rightSize, outputActivation);
				newWeightsDelta[j] = 0;
				newMomentums[j] = 0;
				newVariances[j] = 0;
			}
			newLayerWeights[i] = newWeights;
			newLayerWeightsDelta[i] = newWeightsDelta;
			newLayerMomentums[i] = newMomentums;
			newLayerVariances[i] = newVariances;
		}
		weight.push_back(newLayerWeights);
		weightDelta.push_back(newLayerWeightsDelta);
		momentum.push_back(newLayerMomentums);
		variance.push_back(newLayerVariances);
	}
}

void Network::CopyWeights(Network* source) {
	//Copy Biases
	for (int i = 0; i < hiddenNumber; i++)
		for (int j = 0; j < hiddenSize; j++)
			hidden[i][j].bias = source->hidden[i][j].bias;

	for (int i = 0; i < outputSize; i++)
		output[i].bias = source->output[i].bias;

	//Update Weights
	for (int k = 0; k <= hiddenNumber; k++) {
		int leftSize, rightSize;
		if (k == 0) {
			leftSize = inputSize;
			rightSize = hiddenSize;
		}
		else if (k == hiddenNumber) {
			leftSize = hiddenSize;
			rightSize = outputSize;
		}
		else {
			leftSize = hiddenSize;
			rightSize = hiddenSize;
		}
		for (int i = 0; i < leftSize; i++)
			for (int j = 0; j < rightSize; j++)
				weight[k][i][j] = source->weight[k][i][j];
	}
}

vector<float> Network::Propagate(vector<float>& in) {
	this->input = in;

	//Input to Hidden
	//#pragma omp parallel for
	for (int i = 0; i < hiddenSize; i++) {
		float sum = 0;
		for (int j = 0; j < inputSize; j++) {
			sum += input[j] * weight[0][j][i];//Paralelizacija izgleda da je spora, treba malo jos testirati
		}
		sum += hidden[0][i].bias;
		hidden[0][i].in = sum;
		hidden[0][i].out = activation(sum, hiddenActivation);
	}

	//Hidden to Hidden
	for (int k = 1; k < hiddenNumber; k++) {
		for (int i = 0; i < hiddenSize; i++) {
			float sum = 0;
			for (int j = 0; j < hiddenSize; j++) {
				sum += hidden[k - 1][j].out * weight[k][j][i];
			}
			sum += hidden[k][i].bias;
			hidden[k][i].in = sum;
			hidden[k][i].out = activation(sum, hiddenActivation);
		}
	}

	//Hidden to Output
	vector<float> outSoft(outputSize);
	for (int i = 0; i < outputSize; i++) {
		float sum = 0;
		for (int j = 0; j < hiddenSize; j++) {
			sum += hidden[hiddenNumber - 1][j].out * weight[hiddenNumber][j][i];
		}
		sum += output[i].bias;
		output[i].in = sum;
		outSoft[i] = sum;
		if (outputActivation != SOFTMAX)
			output[i].out = activation(sum, outputActivation);
	}
	//Softmax u zadnjem
	if (outputActivation == SOFTMAX) {
		softmax(outSoft);
		for (int i = 0; i < outputSize; i++)
			output[i].out = outSoft[i];
	}

	vector<float> out;
	for (int i = 0; i < outputSize; i++)
		out.push_back(output[i].out);

	return out;
}

float Network::GradientDescent(float gradient) {
	return gradient * learningRate;
}

vector<float> Network::Adam(float gradient, float momentum, float variance) {
	float newMomentum = momentumDecay * momentum + (1 - momentumDecay) * gradient;
	float newVariance = varianceDecay * variance + (1 - varianceDecay) * gradient * gradient;

	float momentumHat = newMomentum / (1 - pow(momentumDecay, iteration + 1));
	float varianceHat = newVariance / (1 - pow(varianceDecay, iteration + 1));

	float out = momentumHat * learningRate / (sqrt(varianceHat) + epsilon);

	vector<float> ret(3);
	ret[0] = out;
	ret[1] = newMomentum;
	ret[2] = newVariance;
	return ret;
}

void Network::Fit(vector<float>& in, vector<float>& correctOutput) {
	vector<float> predicted = Propagate(in);

	int correctIndex = 0;
	for (int i = 0; i < outputSize; i++)
		if (correctOutput[i] == 1) {
			correctIndex = i;
			break;
		}
	if (outputActivation == SOFTMAX) {
		loss += crossEntropyLoss(predicted[correctIndex]);
	}
	//CORRECT HIDDEN TO OUTPUT AND SET OUTPUT GRADIENTS
	for (int i = 0; i < outputSize; i++) {
		float gradient;
		if (outputActivation != SOFTMAX) {
			loss += pow(output[i].out - correctOutput[i], 2);//SSE loss
			gradient = (output[i].out - correctOutput[i]) * activationDerivative(output[i].in, outputActivation);
		}
		else {
			float error = crossEntropyLossDerivative(output[correctIndex].out);
			gradient = error * softmaxDerivative(output[correctIndex].out, output[i].out, i == correctIndex);
		}
		output[i].gradient = gradient;
		output[i].biasDelta += gradient;
		for (int j = 0; j < hiddenSize; j++) {
			weightDelta[hiddenNumber][j][i] += gradient * hidden[hiddenNumber - 1][j].out;
		}
	}
	//CORRECT H-H-O AND SET LAST HIDDEN GRADIENTS
	if (hiddenNumber > 1) {
		for (int i = 0; i < hiddenSize; i++) {
			float gradientSum = 0;
			for (int j = 0; j < outputSize; j++) {
				gradientSum += output[j].gradient * weight[hiddenNumber][i][j];
			}
			float gradient = gradientSum * activationDerivative(hidden[hiddenNumber - 1][i].in, hiddenActivation);
			hidden[hiddenNumber - 1][i].gradient = gradient;
			hidden[hiddenNumber - 1][i].biasDelta += gradient;
			for (int j = 0; j < hiddenSize; j++) {
				hidden[hiddenNumber - 1][i].biasDelta += gradient * hidden[hiddenNumber - 2][j].out;
			}
		}
	}
	//CORRECT HIDDEN TO HIDDEN AND SET HIDDEN GRADIENTS
	if (hiddenNumber > 2) {
		for (int k = hiddenNumber - 2; k > 0; k--) {
			for (int i = 0; i < hiddenSize; i++) {
				float gradientSum = 0;
				for (int j = 0; j < hiddenSize; j++) {
					gradientSum += hidden[k + 1][j].gradient * weight[k + 1][i][j];
				}
				float gradient = gradientSum * activationDerivative(hidden[k][i].in, hiddenActivation);
				hidden[k][i].gradient = gradient;
				hidden[k][i].biasDelta += gradient;
				for (int j = 0; j < hiddenSize; j++) {
					weightDelta[k][j][i] += gradient * hidden[k - 1][j].out;
				}
			}
		}
	}
	//CORRECT INPUT TO HIDDEN; IF HIDDENSIZE > 1
	//#pragma omp parallel for
	for (int i = 0; i < hiddenSize; i++) {
		float gradientSum = 0;
		for (int j = 0; j < hiddenSize; j++) {
			gradientSum += hidden[1][j].gradient * weight[1][i][j];
		}
		float gradient = gradientSum * activationDerivative(hidden[0][i].in, hiddenActivation);
		hidden[0][i].biasDelta += gradient;
		for (int j = 0; j < inputSize; j++) {
			weightDelta[0][j][i] += gradient * input[j];
		}
	}
	batchCount++;
}

void Network::UpdateWeights() {
	vector<float> adamOut;
	//Update Biases
	for (int i = 0; i < hiddenNumber; i++) {
		for (int j = 0; j < hiddenSize; j++) {
			float gradient = hidden[i][j].biasDelta / batchCount;

			float currentMomentum = hidden[i][j].momentum;
			float currentVariance = hidden[i][j].variance;

			adamOut = Adam(gradient, currentMomentum, currentVariance);

			//hidden[i][j].bias -= GradientDescent(gradient);
			hidden[i][j].bias -= adamOut[0];
			hidden[i][j].momentum = adamOut[1];
			hidden[i][j].variance = adamOut[2];

			hidden[i][j].biasDelta = 0;
		}
	}
	for (int i = 0; i < outputSize; i++) {
		float gradient = output[i].biasDelta / batchCount;

		float currentMomentum = output[i].momentum;
		float currentVariance = output[i].variance;

		adamOut = Adam(gradient, currentMomentum, currentVariance);

		//output[i].bias -= GradientDescent(gradient);
		output[i].bias -= adamOut[0];
		output[i].momentum = adamOut[1];
		output[i].variance = adamOut[2];

		output[i].biasDelta = 0;
	}
	//Update Weights
	for (int k = 0; k <= hiddenNumber; k++) {
		int leftSize, rightSize;
		if (k == 0) {
			leftSize = inputSize;
			rightSize = hiddenSize;
		}
		else if (k == hiddenNumber) {
			leftSize = hiddenSize;
			rightSize = outputSize;
		}
		else {
			leftSize = hiddenSize;
			rightSize = hiddenSize;
		}
		for (int i = 0; i < leftSize; i++) {
			for (int j = 0; j < rightSize; j++) {
				float gradient = weightDelta[k][i][j] / batchCount;

				float currentMomentum = momentum[k][i][j];
				float currentVariance = variance[k][i][j];

				adamOut = Adam(gradient, currentMomentum, currentVariance);

				//weight[k][i][j] -= GradientDescent(gradient);
				weight[k][i][j] -= adamOut[0];
				momentum[k][i][j] = adamOut[1];
				variance[k][i][j] = adamOut[2];

				weightDelta[k][i][j] = 0;
			}
		}
	}
	batchCount = 0;
	iteration++;
}

float Network::resetLoss() {
	float ret = loss;
	loss = 0;
	return ret;
}

void Network::setLearningRate(float learningRate) {
	this->learningRate = learningRate;
}

Network::~Network()
{
}
