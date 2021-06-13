#pragma once
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct BufferSample {
	vector<vector<float>> states;
	vector<vector<float>> actions;
	vector<float> rewards;
	vector<vector<float>> newStates;
	vector<float> terminal;
};

class ReplayBuffer
{
private:
	int memorySize;
	
	int actionNumber;
	bool discrete = true;
	vector<vector<float>> stateMemory;
	vector<vector<float>> newStateMemory;
	vector<vector<float>> actionMemory;
	vector<float> rewardMemory;
	vector<float> terminalMemory;

	vector<int> chooseBatch(int batchSize);
public:
	int memoryCounter;
	ReplayBuffer(int maxSize, int inputSize, int actionNumber); //Assume discrete
	void StoreTransition(vector<float> state, int action, float reward, vector<float> newState, bool done);
	BufferSample SampleBuffer(int batchSize);
	~ReplayBuffer();
};

