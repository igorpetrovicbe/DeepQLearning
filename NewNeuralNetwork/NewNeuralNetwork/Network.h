#pragma once
#include <iostream>
#include <vector>
#include <time.h>
#include <conio.h>
#include <iomanip>
#include <omp.h>

using namespace std;

struct Neuron {
	float in;
	float out;
	float gradient;
	float bias;
	float biasDelta;
	float momentum;
	float variance;
};

enum Activation { SIGMOID, RELU, SOFTMAX, NONE };

class Network
{
private:
	int inputSize;
	int hiddenSize;
	int hiddenNumber;
	int outputSize;
	float batchCount;
	float learningRate;
	float momentumDecay = 0.9;
	float varianceDecay = 0.999;
	float epsilon = 0.00000001;
	int iteration = 0;
	Activation hiddenActivation = RELU;
	Activation outputActivation = NONE;
	float loss = 0;
	
	vector<float> input;
	vector<Neuron> output;
	vector<vector<Neuron>> hidden;

	vector<vector<vector<float>>> weight;
	vector<vector<vector<float>>> weightDelta;
	vector<vector<vector<float>>> momentum;
	vector<vector<vector<float>>> variance;

	float GradientDescent(float gradient);
	vector<float> Adam(float gradient, float momentum, float variance);
public:
	Network(int, int, int, int, float);
	void CopyWeights(Network* source);
	vector<float> Propagate(vector<float>&);
	void Fit(vector<float>&, vector<float>&);
	void UpdateWeights();
	float resetLoss();
	void setLearningRate(float);
	~Network();
};

