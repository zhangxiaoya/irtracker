#pragma once
#include <cstdint>

struct WarningBox
{
	WarningBox() : blockX(-1), blockY(-1), timeLeft(0)
	{
	}

	int blockX;
	int blockY;
	uint32_t timeLeft;
};

const auto MaxWarningBoxCount = 2;

WarningBox GlobalWarningBoxs[MaxWarningBoxCount];

const auto TimeLeftLimie = 5;