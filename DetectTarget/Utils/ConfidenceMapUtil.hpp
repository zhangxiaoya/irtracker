#pragma once
#include <vector>
#include <core/core.hpp>
#include "../Models/ConfidenceElem.hpp"

class ConfidenceMapUtil
{
public:

	static bool CheckIfInTopCount(const cv::Rect& rect, int searchIndex, const std::vector<ConfidenceElem>& confidenceElems);

	static void LostMemory(int queueSize, int& currentIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap);

private:

	const static int CONFIDENCE_MIN_THRESHOLD = 20;
};

inline bool ConfidenceMapUtil::CheckIfInTopCount(const cv::Rect& rect, int searchIndex, const std::vector<ConfidenceElem>& confidenceElems)
{
	auto x = (rect.x + rect.width / 2) / BLOCK_SIZE;
	auto y = (rect.y + rect.height / 2) / BLOCK_SIZE;

	for (auto i = 0; i < searchIndex; ++i)
	{
		if (confidenceElems[i].x == x && confidenceElems[i].y == y && confidenceElems[i].confidenceVal >= CONFIDENCE_MIN_THRESHOLD)
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
