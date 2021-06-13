#pragma once
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
//#include "rlutil.h"

enum Action { RIGHT, UP, LEFT, DOWN };

struct Point {
	int x;
	int y;
};

using namespace std;

class EnvironmentMonster
{
private:
	const int length = 5;
	float score;
	float reward;
	bool done;
	Point startPosition;
	Point position;
	Point rewardPosition;
	int steps = 0;
	const int maxSteps = 20;

	const float goalReward = 100;
	const float stepPunishment = 1;

	void Move(Action action);
public:
	EnvironmentMonster(Point startPos);
	vector<float> Step(Action action);
	vector<float> Reset();
	vector<float> Observe();
	void PrintObservation();
	float getReward();
	float getScore();
	bool getDone();
};