#pragma once
#include <cstdint>
#include <core/core.hpp>

struct WarningBox
{
	WarningBox() : blockX(-1), blockY(-1), timeLeft(0), targetRect(0, 0, 0, 0)
	{
	}

	int blockX;
	int blockY;
	uint32_t timeLeft;
	cv::Rect targetRect;

};

const auto MaxWarningBoxCount = 2;

WarningBox GlobalWarningBoxs[MaxWarningBoxCount];

const auto TimeLeftLimie = 5;