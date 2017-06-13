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
	uint32_t timeLeft;
	cv::Rect targetRect;
	int leftTopX;
	int leftTopY;

};

const auto MaxWarningBoxCount = 2;

TargetTracker GlobalWarningBoxs[MaxWarningBoxCount];

const auto TimeLeftLimie = 5;