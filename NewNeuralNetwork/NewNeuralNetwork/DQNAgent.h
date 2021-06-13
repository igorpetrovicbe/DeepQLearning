#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include "ReplayBuffer.h"
#include "Network.h"

using namespace std;

class DQNAgent
{
private:
	float gamma;
	float epsilon;
	int actionNumber;
	float epsilonDecrease = 0.996;
	float epsilonMin = 0.01;
	int memorySize = 1000000;
	int batchSize;
	ReplayBuffer* memory;
	Network* mainNet;
	Network* targetNet;

	vector<float> actionSpace;
public:
	DQNAgent(float learningRate, float gamma, int actionNumber, float epsilon, int batchSize, int inputSize, float epsilonEnd, int memorySize);
	void Remember(vector<float> state, int action, float reward, vector<float> newState, bool done);
	int ChooseAction(vector<float> state);
	int ChooseActionExploit(vector<float> state);
	void UpdateTarget();
	void Learn();
	~DQNAgent();
};