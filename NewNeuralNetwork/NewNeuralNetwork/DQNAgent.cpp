#include "stdafx.h"
#include "DQNAgent.h"

bool choice(float probability) {
	float r = rand() % 10000;
	return r / 10000 < probability;
}

DQNAgent::DQNAgent(float learningRate, float gamma, int actionNumber, float epsilon, int batchSize, int inputSize, float epsilonEnd, int memorySize) {
	actionSpace = vector<float>(actionNumber);
	this->actionNumber = actionNumber;
	this->gamma = gamma;
	this->epsilon = epsilon;
	this->epsilonDecrease = epsilonDecrease;
	this->epsilonMin = epsilonEnd;
	this->batchSize = batchSize;

	int hiddenSize = 10;
	int hiddenNumber = 2;

	memory = new ReplayBuffer(memorySize, inputSize, actionNumber);
	mainNet = new Network(inputSize, hiddenSize, hiddenNumber, actionNumber, learningRate);
	targetNet = new Network(inputSize, hiddenSize, hiddenNumber, actionNumber, learningRate);
	targetNet->CopyWeights(mainNet);
}

void DQNAgent::Remember(vector<float> state, int action, float reward, vector<float> newState, bool done) {
	memory->StoreTransition(state, action, reward, newState, done);
}

int DQNAgent::ChooseAction(vector<float> state) {
	int action;
	if (choice(epsilon)) {
		action = rand() % actionNumber;
	}
	else {
		vector<float> actions = targetNet->Propagate(state);
		action = 0;
		for (int i = 0; i < actionNumber; i++)
			if (actions[i] > actions[action])
				action = i;
	}
	return action;
}

int DQNAgent::ChooseActionExploit(vector<float> state) {
	int action;
	vector<float> actions = targetNet->Propagate(state);
	action = 0;
	cout << "(";
	for (int i = 0; i < actionNumber; i++) {
		cout << actions[i] << ", ";
		if (actions[i] > actions[action])
			action = i;
	}
	cout << ")" << endl;
	return action;
}

void DQNAgent::Learn() {
	if (memory->memoryCounter < batchSize)
		return;
	BufferSample batch = memory->SampleBuffer(batchSize);

	vector<float> actionValues(actionNumber);
	vector<float> qEvaluate(actionNumber);
	vector<float> qNext(actionNumber);
	for (int i = 0; i < batchSize; i++) {
		qEvaluate = targetNet->Propagate(batch.states[i]);
		qNext = targetNet->Propagate(batch.newStates[i]);

		int action = 0;
		for (int j = 0; j < actionNumber; j++)
			if (batch.actions[i][j] > batch.actions[i][action])
				action = j;

		int bestNext = qNext[0];
		for (int j = 0; j < actionNumber; j++)
			if (qNext[j] > bestNext)
				bestNext = qNext[j];

		qEvaluate[action] = batch.rewards[i] + gamma * bestNext * batch.terminal[i];

		mainNet->Fit(batch.states[i], qEvaluate);
	}

	mainNet->UpdateWeights();

	if (epsilon > epsilonMin)
		epsilon = epsilon * epsilonDecrease;
	else
		epsilon = epsilonMin;
}

void DQNAgent::UpdateTarget() {
	targetNet->CopyWeights(mainNet);
}

DQNAgent::~DQNAgent()
{
}