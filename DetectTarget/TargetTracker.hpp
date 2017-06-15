#pragma once
#include <core/core.hpp>

class TargetTracker
{

public:

	TargetTracker() : blockX(-1), blockY(-1), timeLeft(0), targetRect(0, 0, 0, 0), leftTopX(-1), leftTopY(-1), warningStageTime(5), maxLifeTime(100)
	{
	}

	cv::Scalar Color() const
	{
		
		if (timeLeft >= warningStageTime)
			return REDCOLOR;
		return YELLOWCOLOR;
	}

	void ExtendLifeTime()
	{
		if (timeLeft < maxLifeTime)
			timeLeft++;
	}

	int blockX;
	int blockY;
	int timeLeft;
	cv::Rect targetRect;
	int leftTopX;
	int leftTopY;

private:
	int warningStageTime;
	int maxLifeTime;
};

const auto MaxTrackingTargetCount = 2;
const auto TimeLeftLimit = 5;

std::vector<TargetTracker> GlobalTrackerList;