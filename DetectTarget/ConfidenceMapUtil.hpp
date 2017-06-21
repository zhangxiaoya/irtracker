#pragma once
#include "ConfidenceElem.hpp"
#include <vector>
#include <core/core.hpp>
#include "GlobalConstantConfigure.h"

class ConfidenceMapUtil
{
public:

	static bool CheckIfInTopCount(const cv::Rect& rect, int searchIndex, const std::vector<ConfidenceElem>& confidenceElems);

	static void LostMemory(int queueSize, int& currentIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap);
};

inline bool ConfidenceMapUtil::CheckIfInTopCount(const cv::Rect& rect, int searchIndex, const std::vector<ConfidenceElem>& confidenceElems)
{
	auto x = (rect.x + rect.width / 2) / STEP;
	auto y = (rect.y + rect.height / 2) / STEP;

	for (auto i = 0; i < searchIndex; ++i)
	{
		if (confidenceElems[i].x == x && confidenceElems[i].y == y && confidenceElems[i].confidenceVal >= 40)
			return true;
	}
	return false;
}

inline void ConfidenceMapUtil::LostMemory(int queueSize, int& currentIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap)
{
	currentIndex++;
	currentIndex %= queueSize;
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			confidenceMap[y][x][currentIndex] = 0;
		}
	}
}
