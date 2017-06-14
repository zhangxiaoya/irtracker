#pragma once
#include <cstdint>
#include <core/core.hpp>

struct TargetTracker
{
	TargetTracker() : blockX(-1), blockY(-1), timeLeft(0), targetRect(0, 0, 0, 0), leftTopX(-1), leftTopY(-1)
	{
	}

	int blockX;
	int blockY;
	int timeLeft;
	cv::Rect targetRect;
	int leftTopX;
	int leftTopY;

};

const auto MaxTrackingTargetCount = 2;
const auto TimeLeftLimit = 5;

std::vector<TargetTracker> GlobalTrackerList;