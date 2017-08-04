#pragma once
#include <core/core.hpp>

class TargetTracker
{

public:

	TargetTracker() : blockX(-1), blockY(-1), timeLeft(0), targetRect(0, 0, 0, 0), leftTopX(-1), leftTopY(-1), warningStageTime(5), maxLifeTime(20)
	{
	}

	cv::Scalar Color() const
	{
		if (timeLeft >= warningStageTime)
			return COLOR_RED;
		return COLOR_YELLOW;
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
	std::vector<uchar> feature;

private:
	int warningStageTime;
	int maxLifeTime;
};

const auto MaxTrackingTargetCount = 2;
const auto TimeLeftLimit = 5;

std::vector<TargetTracker> GlobalTrackerList;