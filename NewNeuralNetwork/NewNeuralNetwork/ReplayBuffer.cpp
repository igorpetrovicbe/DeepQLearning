#include "stdafx.h"
#include "ReplayBuffer.h"

int randomRange(int min, int max) {
	int r = rand() % (max - min);
	return r + min;
}

ReplayBuffer::ReplayBuffer(int maxSize, int inputSize, int actionNumber) {
	memorySize = maxSize;
	for (int i = 0; i < maxSize; i++) {
		vector<float> state(inputSize);
		vector<float> newState(inputSize);
		vector<float> action(actionNumber);
		stateMemory.push_back(state);
		newStateMemory.push_back(newState);
		actionMemory.push_back(action);
	}
	rewardMemory = vector<float>(memorySize);
	terminalMemory = vector<float>(memorySize);
	this->actionNumber = actionNumber;
}

void ReplayBuffer::StoreTransition(vector<float> state, int action, float reward, vector<float> newState, bool done) { //TODO: non-discrete action
	int index = memoryCounter % memorySize;
	stateMemory[index] = state;
	newStateMemory[index] = newState;
	rewardMemory[index] = reward;
	terminalMemory[index] = 1 - (float)done;

	vector<float> actions(actionNumber);
	actions[action] = 1;
	actionMemory[index] = actions; //Assume discrete

	memoryCounter++;
}

vector<int> ReplayBuffer::chooseBatch(int batchSize) {
	int maxMemory = min(memoryCounter, memorySize);
	vector<int> batch(batchSize);
	int filled = 0;
	while (filled < batchSize) {
		int r = rand() % maxMemory;
		bool exists = false;
		for (int i = 0; i < filled; i++) {
			if (batch[i] == r) {
				exists = true;
				break;
			}
		}
		if (!exists) {
			batch[filled] = r;
			filled++;
		}
	}
	return batch;
}

BufferSample ReplayBuffer::SampleBuffer(int batchSize) {
	vector<int> batch = chooseBatch(batchSize);

	BufferSample sample;
	for (int i = 0; i < batchSize; i++) {
		sample.states.push_back(stateMemory[batch[i]]);
		sample.actions.push_back(actionMemory[batch[i]]);
		sample.rewards.push_back(rewardMemory[batch[i]]);
		sample.newStates.push_back(newStateMemory[batch[i]]);
		sample.terminal.push_back(terminalMemory[batch[i]]);
	}
	return sample;
}

ReplayBuffer::~ReplayBuffer()
{
}
