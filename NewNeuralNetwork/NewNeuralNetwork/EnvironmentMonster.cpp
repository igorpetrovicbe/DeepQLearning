#include "stdafx.h"
#include "EnvironmentMonster.h"

using namespace std;

EnvironmentMonster::EnvironmentMonster(Point pos) {
	startPosition = pos;
	position = pos;
	rewardPosition.x = length - 1;
	rewardPosition.y = length - 1;

	done = false;
	reward = 0;
	score = 0;
}

void EnvironmentMonster::Move(Action action) {
	if (action == RIGHT && position.x < length - 1) {
		position.x++;
	}
	else if (action == LEFT && position.x > 0) {
		position.x--;
	}
	else if (action == DOWN && position.y < length - 1) {
		position.y++;
	}
	else if (action == UP && position.y > 0) {
		position.y--;
	}
}

vector<float> EnvironmentMonster::Step(Action action) {
	Move(action);

	steps++;

	if (steps == maxSteps)
		done = true;

	if (position.x == rewardPosition.x && position.y == rewardPosition.y) {
		reward = goalReward;
		score += goalReward;
		done = true;
	}
	else {
		reward = -stepPunishment;
		score += reward;
	}

	return Observe();
}

vector<float> EnvironmentMonster::Observe() {
	vector<float> ret(length*length);
	for (int i = 0; i < length; i++)
		for (int j = 0; j < length; j++)
			if (position.x == j && position.y == i)
				ret[i*length + j] = 1;
			else
				ret[i*length + j] = 0;
	return ret;
}

void EnvironmentMonster::PrintObservation() {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < length; j++)
			if (position.x == j && position.y == i)
				cout << "P ";
			else
				cout << ". ";
		cout << endl;
	}
}

vector<float> EnvironmentMonster::Reset() {
	steps = 0;
	reward = 0;
	score = 0;
	done = false;
	position = startPosition;

	return Observe();
}

bool EnvironmentMonster::getDone() {
	return done;
}

float EnvironmentMonster::getReward() {
	return reward;
}

float EnvironmentMonster::getScore() {
	return score;
}