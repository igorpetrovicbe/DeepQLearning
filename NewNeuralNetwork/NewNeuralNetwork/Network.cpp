#include "stdafx.h"
#include "Network.h"

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float activation(float x) {
	return sigmoid(x);
}

float activation_derivative(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
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
	for (int i = 0; i < hiddenNumber; i++) {//TODO napraviti bolju inicijalizaciju tezina i ulepsati kod
		vector<Neuron> newHidden(hiddenSize);
		for (int j = 0; j < hiddenSize; j++) {
			Neuron neuron;
			r = rand() % 10000;
			neuron.bias = (r - 5000) / 5000;
			neuron.out = 0;
			neuron.gradient = 0;
			neuron.biasDelta = 0;
			neuron.momentum = 0;
			neuron.variance = 0;
			newHidden[j] = neuron;
		}
		hidden.push_back(newHidden);
	}
	for (int i = 0; i < outputSize; i++) {
		Neuron neuron;
		r = rand() % 10000;
		neuron.bias = (r - 5000) / 5000;
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
				r = rand() % 10000;
				newWeights[j] = (r - 5000) / 5000;
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

vector<float> Network::Propagate(vector<float>& in) {
	this->input = in;

	//Input to Hidden
	for (int i = 0; i < hiddenSize; i++) {
		float sum = 0;
		for (int j = 0; j < inputSize; j++) {
			sum += input[j] * weight[0][j][i];
		}
		sum += hidden[0][i].bias;
		hidden[0][i].in = sum;
		hidden[0][i].out = activation(sum);
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
			hidden[k][i].out = activation(sum);
		}
	}

	//Hidden to Output
	for (int i = 0; i < outputSize; i++) {
		float sum = 0;
		for (int j = 0; j < hiddenSize; j++) {
			sum += hidden[hiddenNumber - 1][j].out * weight[hiddenNumber][j][i];
		}
		sum += output[i].bias;
		output[i].in = sum;
		output[i].out = activation(sum);
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
	//CORRECT HIDDEN TO OUTPUT AND SET OUTPUT GRADIENTS
	for (int i = 0; i < outputSize; i++) {
		float gradient = (output[i].out - correctOutput[i]) * activation_derivative(output[i].in);
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
			float gradient = gradientSum * activation_derivative(hidden[hiddenNumber - 1][i].in);
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
				float gradient = gradientSum * activation_derivative(hidden[k][i].in);
				hidden[k][i].gradient = gradient;
				hidden[k][i].biasDelta += gradient;
				for (int j = 0; j < hiddenSize; j++) {
					weightDelta[k][j][i] += gradient * hidden[k - 1][j].out;
				}
			}
		}
	}
	//CORRECT INPUT TO HIDDEN; IF HIDDENSIZE > 1
	for (int i = 0; i < hiddenSize; i++) {
		float gradientSum = 0;
		for (int j = 0; j < hiddenSize; j++) {
			gradientSum += hidden[1][j].gradient * weight[1][i][j];
		}
		float gradient = gradientSum * activation_derivative(hidden[0][i].in);
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

Network::~Network()
{
}
