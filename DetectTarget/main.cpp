#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "Models/FieldType.hpp"
#include "Headers/GlobalConstantConfigure.h"
#include "Utils/ConfidenceMapUtil.hpp"
#include "Tracker/TargetTracker.hpp"
#include "Detector/DetectByMaxFilterAndAdptiveThreshold.hpp"
#include "Utils/SpecialUtil.hpp"
#include <imgproc/types_c.h>
#include <imgproc/imgproc.hpp>
#include "Utils/Util.hpp"
#include "Models/ConfidenceElem.hpp"
#include "Models/DrawResultType.hpp"

void UpdateConfidenceQueueMap(int queueEndIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap, const std::vector<cv::Rect>& targetRects, FieldType fieldType = Four)
{
	std::vector<std::vector<bool>> updateFlag(countY, std::vector<bool>(countX, false));

	const auto fourIncrement = 2;
	const auto eightIncrement = 1;

	for (auto i = 0; i < targetRects.size(); ++i)
	{
		auto rect = targetRects[i];
		auto x = (rect.x + rect.width / 2) / BLOCK_SIZE;
		auto y = (rect.y + rect.height / 2) / BLOCK_SIZE;

//		if (updateFlag[y][x])
//			continue;
		// center
		confidenceMap[y][x][queueEndIndex] += 10;
		updateFlag[y][x] = true;
		// up
		confidenceMap[y - 1 >= 0 ? y - 1 : 0][x][queueEndIndex] += fourIncrement;
		// down
		confidenceMap[y + 1 < countY ? y + 1 : countY - 1][x][queueEndIndex] += fourIncrement;
		// left
		confidenceMap[y][x - 1 > 0 ? x - 1 : 0][queueEndIndex] += fourIncrement;
		// right
		confidenceMap[y][x + 1 < countX ? x + 1 : countX - 1][queueEndIndex] += fourIncrement;

		if (fieldType == Eight)
		{
			// up left
			confidenceMap[y - 1 >= 0 ? y - 1 : 0][x - 1 >= 0 ? x - 1 : 0][queueEndIndex] += eightIncrement;
			// down right
			confidenceMap[y + 1 < countY ? y + 1 : countY - 1][x + 1 < countX ? x + 1 : countX - 1][queueEndIndex] += eightIncrement;
			// left down
			confidenceMap[y + 1 < countY ? y + 1 : countY - 1][x - 1 > 0 ? x - 1 : 0][queueEndIndex] += eightIncrement;
			// right up
			confidenceMap[y - 1 >= 0 ? y - 1 : 0][x + 1 < countX ? x + 1 : countX - 1][queueEndIndex] += eightIncrement;
		}
	}
}

void UpdateRectLayoutMatrix(std::vector<std::vector<int>>& rectLayoutMatrix, const std::vector<cv::Rect>& targetRects, FieldType fieldType = Four)
{
	std::vector<std::vector<bool>> updateFlag(countY, std::vector<bool>(countX, false));

	const auto fourIncrement = 2;
	const auto eightIncrement = 1;

	for (auto i = 0; i < targetRects.size(); ++i)
	{
		auto rect = targetRects[i];
		auto x = (rect.x + rect.width / 2) / BLOCK_SIZE;
		auto y = (rect.y + rect.height / 2) / BLOCK_SIZE;

		if (updateFlag[y][x])
			continue;
		// center
		rectLayoutMatrix[y][x] = 10;
		updateFlag[y][x] = true;
		// up
		rectLayoutMatrix[y - 1 >= 0 ? y - 1 : 0][x] += fourIncrement;
		// down
		rectLayoutMatrix[y + 1 < countY ? y + 1 : countY - 1][x] += fourIncrement;
		// left
		rectLayoutMatrix[y][x - 1 > 0 ? x - 1 : 0] += fourIncrement;
		// right
		rectLayoutMatrix[y][x + 1 < countX ? x + 1 : countX - 1] += fourIncrement;

		if (fieldType == Eight)
		{
			// up left
			rectLayoutMatrix[y - 1 >= 0 ? y - 1 : 0][x - 1 >= 0 ? x - 1 : 0] += eightIncrement;
			// down right
			rectLayoutMatrix[y + 1 < countY ? y + 1 : countY - 1][x + 1 < countX ? x + 1 : countX - 1] += eightIncrement;
			// left down
			rectLayoutMatrix[y + 1 < countY ? y + 1 : countY - 1][x - 1 > 0 ? x - 1 : 0] += eightIncrement;
			// right up
			rectLayoutMatrix[y - 1 >= 0 ? y - 1 : 0][x + 1 < countX ? x + 1 : countX - 1] += eightIncrement;
		}
	}
}

void UpdateVectorOfConfidenceQueueMap(const std::vector<std::vector<std::vector<int>>>& confidenceQueueMap, std::vector<ConfidenceElem>& vectorOfConfidenceQueueMap)
{
	auto confidenceIndex = 0;
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			vectorOfConfidenceQueueMap[confidenceIndex].x = x;
			vectorOfConfidenceQueueMap[confidenceIndex].y = y;
			vectorOfConfidenceQueueMap[confidenceIndex++].confidenceVal = Util::Sum(confidenceQueueMap[y][x]);
		}
	}

	sort(vectorOfConfidenceQueueMap.begin(), vectorOfConfidenceQueueMap.end(), Util::CompareConfidenceValue);
}

void GetMostLiklyTargetsRect(const std::vector<ConfidenceElem>& vectorOfConfidenceQueueMap, int& searchIndex)
{
	auto currentTopCount = 0;

	while (searchIndex < vectorOfConfidenceQueueMap.size())
	{
		if (searchIndex == 0)
		{
			++currentTopCount;
			++searchIndex;
			continue;
		}
		if (vectorOfConfidenceQueueMap[currentTopCount].confidenceVal == vectorOfConfidenceQueueMap[searchIndex].confidenceVal)
		{
			++searchIndex;
		}
		else
		{
			++currentTopCount;
			++searchIndex;
			if (currentTopCount >= TOP_COUNT_OF_BLOCK_WITH_HIGH_QUEUE_VALUE)
				break;
		}
	}
}

void DrawRectangleForAllDetectedTargetsAndUpdateBlockConfidence(cv::Mat& colorFrame,
                                                                const int searchIndex,
                                                                const std::vector<ConfidenceElem>& vectorOfConfidenceQueuqMap,
                                                                std::vector<cv::Rect>& targetRects,
                                                                std::vector<std::vector<int>>& confidenceValueMap,
																FieldType fieldType = FieldType::Four)
{
	std::vector<std::vector<bool>> updateFlag(countY, std::vector<bool>(countX, false));

	for (auto it = targetRects.begin(); it != targetRects.end(); ++it)
	{
		auto rect = *it;
		if (ConfidenceMapUtil::CheckIfInTopCount(rect, searchIndex, vectorOfConfidenceQueuqMap))
		{
			rectangle(colorFrame, cv::Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2), COLOR_BLUE);

			auto x = (rect.x + rect.width / 2) / BLOCK_SIZE;
			auto y = (rect.y + rect.height / 2) / BLOCK_SIZE;

			if (updateFlag[y][x])
				continue;
			confidenceValueMap[y][x] += 5;
			updateFlag[y][x] = true;
			// neighbor effect
			if (x - 1 >= 0)
				confidenceValueMap[y][x - 1] += 4;
			if (x + 1 < countX)
				confidenceValueMap[y][x + 1] += 4;
			if (y - 1 >= 0)
				confidenceValueMap[y - 1][x] += 4;
			if (y + 1 < countY)
				confidenceValueMap[y + 1][x] += 4;

			if(fieldType == FieldType::Eight)
			{
				if (x - 1 >= 0 && y -1 >= 0)
					confidenceValueMap[y - 1][x - 1] += 4;
				if (x + 1 < countX && y + 1 < countY)
					confidenceValueMap[y + 1][x + 1] += 4;
				if (y - 1 >= 0 && x + 1 < countX)
					confidenceValueMap[y - 1][x + 1] += 4;
				if (y + 1 < countY && x - 1 <= 0)
					confidenceValueMap[y + 1][x - 1] += 4;
			}
		}
		else
		{
			it = targetRects.erase(it);
			if (it == targetRects.end())
				break;
		}
	}
}

void WriteLastResultToDisk(const cv::Mat& colorFrame, const int frameIndex, char writeFileName[])
{
	sprintf_s(writeFileName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalWriteFileNameFormat, frameIndex);
	imwrite(writeFileName, colorFrame);
}

int MaxNeighbor(const std::vector<std::vector<int>>& confidenceValueMap, int y, int x)
{
	auto maxResult = 0;
	if (x - 1 >= 0 && confidenceValueMap[y][x - 1] > maxResult)
		maxResult = confidenceValueMap[y][x - 1];
	if (x + 1 < countX && confidenceValueMap[y][x + 1] > maxResult)
		maxResult = confidenceValueMap[y][x + 1];
	if (y - 1 >= 0 && confidenceValueMap[y - 1][x] > maxResult)
		maxResult = confidenceValueMap[y - 1][x];
	if (y + 1 < countY && confidenceValueMap[y + 1][x] > maxResult)
		maxResult = confidenceValueMap[y + 1][x];
	return maxResult;
}

int MinNeighbor(const std::vector<std::vector<int>>& confidenceValueMap, int y, int x)
{
	auto minResult = 10000;
	if (x - 1 >= 0 && confidenceValueMap[y][x - 1] < minResult)
		minResult = confidenceValueMap[y][x - 1];
	if (x + 1 < countX && confidenceValueMap[y][x + 1] < minResult)
		minResult = confidenceValueMap[y][x + 1];
	if (y - 1 >= 0 && confidenceValueMap[y - 1][x] < minResult)
		minResult = confidenceValueMap[y - 1][x];
	if (y + 1 < countY && confidenceValueMap[y + 1][x] < minResult)
		minResult = confidenceValueMap[y + 1][x];
	return minResult;
}

bool UpdateTrackerStatus(const cv::Rect& rect, int blockX, int blockY, int trackerIndex)
{
	if (trackerIndex != 0)
	{
		if (GlobalTrackerList[trackerIndex - 1].blockX != blockX)
			GlobalTrackerList[trackerIndex - 1].blockX = blockX;
		if (GlobalTrackerList[trackerIndex - 1].blockY != blockY)
			GlobalTrackerList[trackerIndex - 1].blockY = blockY;

		GlobalTrackerList[trackerIndex - 1].leftTopX = rect.x;
		GlobalTrackerList[trackerIndex - 1].leftTopY = rect.y;
		GlobalTrackerList[trackerIndex - 1].targetRect = rect;
		GlobalTrackerList[trackerIndex - 1].ExtendLifeTime();
		return true;
	}
	return false;
}

void PrintConfidenceValueMap(const std::vector<std::vector<int>>& confidenceValueMap, char* infoText)
{
	std::cout << infoText << std::endl;
	for (auto y = 0; y < countY; ++y)
	{
		for (auto x = 0; x < countX; ++x)
			std::cout << std::setw(3) << confidenceValueMap[y][x] << " ";

		std::cout << std::endl;
	}
}

void UpdateConfidenceValueVector(const std::vector<std::vector<int>>& confidenceValueMap, std::vector<ConfidenceElem>& vectorOfConfidenceValueMap)
{
	auto index = 0;
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			vectorOfConfidenceValueMap[index].x = x;
			vectorOfConfidenceValueMap[index].y = y;
			vectorOfConfidenceValueMap[index++].confidenceVal = confidenceValueMap[y][x];
		}
	}

	sort(vectorOfConfidenceValueMap.begin(), vectorOfConfidenceValueMap.end(), Util::CompareConfidenceValue);
}

void GetTopCountBlocksWhichContainsTargets(const std::vector<ConfidenceElem>& vectorOFConfidenceValueMap, std::vector<cv::Point>& blocksContainTargets)
{
	auto currentTargetCountIndex = 0;

	for (auto i = 0; i < vectorOFConfidenceValueMap.size(); ++i)
	{
		if (i == 0)
		{
			blocksContainTargets.push_back(cv::Point(vectorOFConfidenceValueMap[i].x, vectorOFConfidenceValueMap[i].y));
			currentTargetCountIndex++;
			continue;
		}
		if (vectorOFConfidenceValueMap[i].confidenceVal > 0 && vectorOFConfidenceValueMap[i].confidenceVal == vectorOFConfidenceValueMap[i - 1].confidenceVal)
		{
			blocksContainTargets.push_back(cv::Point(vectorOFConfidenceValueMap[i].x, vectorOFConfidenceValueMap[i].y));
		}
		else
		{
			if (vectorOFConfidenceValueMap[i].confidenceVal >= 5)
			{
				blocksContainTargets.push_back(cv::Point(vectorOFConfidenceValueMap[i].x, vectorOFConfidenceValueMap[i].y));
				currentTargetCountIndex++;
				if (currentTargetCountIndex >= TOP_COUNT_OF_TARGET_WITH_HIGH_CONFIDENCE_VALUE)
					break;
			}
		}
	}
}

void SearchWhichTrackerForThisBlock(cv::Point blockPos, int& trackerIndex)
{
	for (auto j = 0; j < GlobalTrackerList.size(); ++j)
	{
		if (GlobalTrackerList[j].blockX == blockPos.x && GlobalTrackerList[j].blockY == blockPos.y)
		{
			trackerIndex = j + 1;
			break;
		}
	}
}

void CreateNewTrackerForThisBlock(cv::Point blockPos, cv::Rect rect)
{
	TargetTracker tracker;
	tracker.blockX = blockPos.x;
	tracker.blockY = blockPos.y;
	tracker.leftTopX = rect.x;
	tracker.leftTopY = rect.y;
	tracker.targetRect = rect;
	tracker.timeLeft = 1;

	GlobalTrackerList.push_back(tracker);
}

void ConfidenceValueLost(std::vector<std::vector<int>> confidenceValueMap)
{
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			if (confidenceValueMap[y][x] > 0)
			{
				confidenceValueMap[y][x] -= 3;
				if (confidenceValueMap[y][x] < 0)
					confidenceValueMap[y][x] = 0;
			}
		}
	}
}

bool ReSearchTarget(const cv::Mat& curFrame, TargetTracker& tracker)
{
	auto leftTopBlockX = tracker.blockX - 1 >= 0 ? tracker.blockX - 1 : tracker.blockX;
	auto leftTopBlockY = tracker.blockY - 1 >= 0 ? tracker.blockY - 1 : tracker.blockY;

	auto rightBottomBlockX = tracker.blockX + 1 < countX ? tracker.blockX + 1 : tracker.blockX;
	auto rightBottomBlockY = tracker.blockY + 1 < countY ? tracker.blockY + 1 : tracker.blockY;

	auto leftTopX = leftTopBlockX * BLOCK_SIZE;
	auto leftTopY = leftTopBlockY * BLOCK_SIZE;

	auto rightBottomX = (rightBottomBlockX + 1) * BLOCK_SIZE - 1;
	if (rightBottomX >= IMAGE_WIDTH)
		rightBottomX = IMAGE_WIDTH - 1;
	auto rightBottomY = (rightBottomBlockY + 1) * BLOCK_SIZE - 1;
	if (rightBottomY >= IMAGE_HEIGHT)
		rightBottomY = IMAGE_HEIGHT - 1;

	auto width = tracker.targetRect.width;
	auto height = tracker.targetRect.height;

	auto minDiff = width * height * height * width;
	auto minRow = -1;
	auto minCol = -1;
	std::vector<uchar> minFeature(width * height, 0);

	for (auto row = leftTopY; row + height - 1 <= rightBottomY; ++ row)
	{
		for (auto col = leftTopX; col + width - 1 <= rightBottomX; ++col)
		{
			auto feature = Util::ToFeatureVector(curFrame(cv::Rect(col, row, width, height)));

			auto curDiff = Util::FeatureDiff(feature, tracker.feature);
			if (curDiff < minDiff)
			{
				minDiff = curDiff;
				minRow = row;
				minCol = col;
				minFeature = feature;
			}
		}
	}

	if (minCol != -1)
	{
		tracker.targetRect = cv::Rect(minCol, minRow, width, height);
		tracker.leftTopX = minCol;
		tracker.leftTopY = minRow;
		tracker.blockX = minCol / BLOCK_SIZE;
		tracker.blockY = minRow / BLOCK_SIZE;
		tracker.feature = minFeature;
		return true;
	}
	return false;
}

void DrawResults(cv::Mat colorFrame)
{
	sort(GlobalTrackerList.begin(), GlobalTrackerList.end(), Util::CompareTracker);
	for (auto tracker : GlobalTrackerList)
	{
		if (tracker.timeLeft > 2)
			rectangle(colorFrame, cv::Rect(tracker.targetRect.x - 2, tracker.targetRect.y - 2, tracker.targetRect.width + 4, tracker.targetRect.height + 4), tracker.Color());
	}
}

void PrintTrackersAndBlocksAndRectsLogs(std::vector<cv::Rect> targetRects, std::vector<cv::Point> blocksContainTargets)
{
	std::cout << "All Tracker" << std::endl;
	for (auto tracker : GlobalTrackerList)
	{
		std::cout << "X = " << tracker.blockX << " Y = " << tracker.blockY << " Time Left = " << tracker.timeLeft << std::endl;
	}

	std::cout << "All Blocks" << std::endl;
	for (auto target : blocksContainTargets)
	{
		std::cout << "X = " << target.x << " Y = "<< target.y <<std::endl;
	}

	std::cout << "All Rects" << std::endl;
	for (auto rect : targetRects)
	{
		std::cout << "X = " << (rect.x + rect.width / 2) / BLOCK_SIZE <<" Y = " << (rect.y + rect.height /2 ) / BLOCK_SIZE<<std::endl;
	}
}

void PrintConfidenceQueueMap(std::vector<std::vector<std::vector<int>>> confidenceQueueMap)
{
	for (auto r = 0; r < countY; ++r)
	{
		for (auto c = 0; c < countX; ++c)
		{
			std::cout << std::setw(3) << Util::Sum(confidenceQueueMap[r][c]) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void DrawHalfRectangle(cv::Mat& colorFrame, const int left, const int top, const int right, const int bottom, const cv::Scalar& lineColor)
{
	line(colorFrame, cv::Point(left, top), cv::Point(left, top + 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(left, top), cv::Point(left + 3, top), lineColor, 1, CV_AA);

	line(colorFrame, cv::Point(right, top), cv::Point(right, top + 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(right, top), cv::Point(right - 3, top), lineColor, 1, CV_AA);

	line(colorFrame, cv::Point(left, bottom), cv::Point(left, bottom - 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(left, bottom), cv::Point(left + 3, bottom), lineColor, 1, CV_AA);

	line(colorFrame, cv::Point(right, bottom), cv::Point(right, bottom - 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(right, bottom), cv::Point(right - 3, bottom), lineColor, 1, CV_AA);
}

void DrawResult(cv::Mat& colorFrame, const cv::Rect& rect, DrawResultType drawResultType = DrawResultType::Rectangles)
{
	auto left = rect.x - 2 < 0 ? 0 : rect.x - 2;
	auto top = rect.y - 2 < 0 ? 0 : rect.y - 2;
	auto right = rect.x + rect.width + 1 >= IMAGE_WIDTH ? IMAGE_WIDTH - 1 : rect.x + rect.width + 1;
	auto bottom = rect.y + rect.height + 1 >= IMAGE_HEIGHT ? IMAGE_HEIGHT - 1 : rect.y + rect.height + 1;
	auto lineColor = COLOR_RED;

	switch (drawResultType)
	{
	case DrawResultType::Rectangles:
		{
			rectangle(colorFrame, cv::Rect(left, top, rect.width + 4, rect.height + 4), COLOR_RED);
			break;
		}

	case DrawResultType::HalfRectangle:
		{
			DrawHalfRectangle(colorFrame, left, top, right, bottom, lineColor);

			break;
		}
	case DrawResultType::Target:
		{
			line(colorFrame, cv::Point(left - 6, (top + bottom) / 2), cv::Point(left - 2, (top + bottom) / 2), lineColor, 1, CV_AA);
			line(colorFrame, cv::Point(right + 2, (top + bottom) / 2), cv::Point(right + 6, (top + bottom) / 2), lineColor, 1, CV_AA);

			line(colorFrame, cv::Point((left + right) / 2, top - 6), cv::Point((left + right) / 2, top - 2), lineColor, 1, CV_AA);
			line(colorFrame, cv::Point((left + right) / 2, bottom + 2), cv::Point((left + right) / 2, bottom + 6), lineColor, 1, CV_AA);
			break;
		}
	case DrawResultType::HalfRectangleWithLine:
		{
		DrawHalfRectangle(colorFrame, left, top, right, bottom, lineColor);

		line(colorFrame, cv::Point(0, (top + bottom) / 2), cv::Point(left - 2, (top + bottom) / 2), lineColor, 1, CV_AA);
		line(colorFrame, cv::Point(right + 2, (top + bottom) / 2), cv::Point(IMAGE_WIDTH - 1, (top + bottom) / 2), lineColor, 1, CV_AA);

		line(colorFrame, cv::Point((left + right) / 2, 0), cv::Point((left + right) / 2, top - 2), lineColor, 1, CV_AA);
		line(colorFrame, cv::Point((left + right) / 2, bottom + 2), cv::Point((left + right) / 2, IMAGE_HEIGHT - 1), lineColor, 1, CV_AA);
		break;
		}
	default:
		rectangle(colorFrame, cv::Rect(left, top, rect.width + 4, rect.height + 4), COLOR_RED);
		break;
	}
}

bool CheckOriginalImageSuroundedBox(const cv::Mat& grayFrame, const cv::Rect& rect)
{
	auto centerX = rect.x + rect.width / 2;
	auto centerY = rect.y + rect.height / 2;

	auto surroundingBoxWidth = 3 * rect.width;
	auto surroundingBoxHeight = 3 * rect.height;

	auto boxLeftTopX = centerX - surroundingBoxWidth / 2 >= 0 ? centerX - surroundingBoxWidth / 2 : 0;
	auto boxLeftTopY = centerY - surroundingBoxHeight / 2 >= 0 ? centerY - surroundingBoxHeight / 2 : 0;
	auto boxRightBottomX = centerX + surroundingBoxWidth / 2 < IMAGE_WIDTH ? centerX + surroundingBoxWidth / 2 : IMAGE_WIDTH - 1;
	auto boxRightBottomY = centerY + surroundingBoxHeight / 2 < IMAGE_HEIGHT ? centerY + surroundingBoxHeight / 2 : IMAGE_HEIGHT - 1;

	auto avgValOfSurroundingBox = Util::AverageValue(grayFrame, cv::Rect(boxLeftTopX, boxLeftTopY, boxRightBottomX - boxLeftTopX + 1, boxRightBottomY - boxLeftTopY + 1));
	auto avgValOfCurrentRect = Util::AverageValue(grayFrame, rect);

	auto convexThreshold = avgValOfSurroundingBox + avgValOfSurroundingBox / 17;
	auto concaveThreshold = avgValOfSurroundingBox - avgValOfSurroundingBox / 20;

	if (std::abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < 3)
		return false;

	uchar centerValue = 0;
	Util::CalCulateCenterValue(grayFrame, centerValue, rect);

	if (avgValOfCurrentRect > convexThreshold || avgValOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

bool CheckDecreatizatedImageSuroundedBox(const cv::Mat& fdImg, const struct CvRect& rect)
{
	auto centerX = rect.x + rect.width / 2;
	auto centerY = rect.y + rect.height / 2;

	auto boxLeftTopX = centerX - 2 * rect.width / 2 >= 0 ? centerX - 2 * rect.width / 2 : 0;
	auto boxLeftTopY = centerY - 2 * rect.height / 2 >= 0 ? centerY - 2 * rect.height / 2 : 0;
	auto boxRightBottomX = centerX + 2 * rect.width / 2 < IMAGE_WIDTH ? centerX + 2 * rect.width / 2 : IMAGE_WIDTH - 1;
	auto boxRightBottomY = centerY + 2 * rect.height / 2 < IMAGE_HEIGHT ? centerY + 2 * rect.height / 2 : IMAGE_HEIGHT - 1;

	auto avgValOfSurroundingBox = Util::AverageValue(fdImg, cv::Rect(boxLeftTopX, boxLeftTopY, boxRightBottomX - boxLeftTopX + 1, boxRightBottomY - boxLeftTopY + 1));
	auto avgValOfCurrentRect = Util::AverageValue(fdImg, rect);

	auto convexThreshold = avgValOfSurroundingBox + avgValOfSurroundingBox / 8;
	auto concaveThreshold = avgValOfSurroundingBox - avgValOfSurroundingBox / 10;

	if (std::abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < 3)
		return false;

	uchar centerValue = 0;
	Util::CalCulateCenterValue(fdImg, centerValue, rect);

	if (avgValOfCurrentRect > convexThreshold || avgValOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

bool CheckFourBlock(const cv::Mat& fdImg,  const cv::Rect& rect)
{
	auto curBlockX = rect.x / BLOCK_SIZE;
	auto curBlockY = rect.y / BLOCK_SIZE;

	if (curBlockX - 1 < 0 || curBlockX + 1 > countX || curBlockY - 1 < 0 || curBlockY + 1 > countY)
		return false;

	auto upAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY - 1);
	auto downAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY + 1);

	auto leftAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY - 1);
	auto rightAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY + 1);

	if(abs(static_cast<int>(upAvg) - static_cast<int>(downAvg)) > 8)
		return false;

	if (abs(static_cast<int>(leftAvg) - static_cast<int>(rightAvg)) > 8)
		return false;

	return true;
}

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture;

	InitVideoReader(video_capture);

	cv::Mat curFrame;
	cv::Mat grayFrame;
	cv::Mat colorFrame;

	auto frameIndex = 0;
	static auto queueEndIndex = 0;

	char writeFileName[WRITE_FILE_NAME_BUFFER_SIZE];
	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];

	std::vector<std::vector<std::vector<int>>> confidenceQueueMap(countY, std::vector<std::vector<int>>(countX, std::vector<int>(QUEUE_SIZE, 0)));
	std::vector<std::vector<int>> confidenceValueMap(countY, std::vector<int>(countX, 0));

	std::vector<std::vector<int>> rectLayoutMatrix(countY, std::vector<int>(countX, 0));

	std::vector<ConfidenceElem> vectorOfConfidenceQueueMap(countX * countY);
	std::vector<ConfidenceElem> vectorOfConfidenceValueMap(countX * countY);

	if (video_capture.isOpened())
	{
		std::cout << "Open Image List Success!" << std::endl;

		while (!curFrame.empty() || frameIndex == 0)
		{
//			video_capture >> curFrame;

			sprintf_s(imageFullName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalImageListNameFormat, frameIndex);
			curFrame = cv::imread(imageFullName);

			if (!curFrame.empty())
			{
				Util::ShowImage(curFrame);

				if(SpecialUtil::CheckFrameIsGray(curFrame, grayFrame))
				{
					cvtColor(curFrame, colorFrame, CV_GRAY2BGR);
				}
				else
				{
					colorFrame = curFrame;
				}

//				SpecialUtil::RemoveInvalidPixel(grayFrame);

				cv::Mat fdImg;
				
				auto targetRects = DetectByMaxFilterAndAdptiveThreshold::Detect<uchar>(grayFrame, fdImg);

//				UpdateRectLayoutMatrix(rectLayoutMatrix, targetRects);

				for (auto rect : targetRects)
				{
					if (
						(
						(CHECK_ORIGIN_FLAG && CheckOriginalImageSuroundedBox(grayFrame, rect)) ||
						(CHECK_DECRETIZATED_FLAG && CheckDecreatizatedImageSuroundedBox(fdImg, rect))
						)
						&&
						CheckFourBlock(fdImg,rect)
						)
					{
						DrawResult(colorFrame, rect, DrawResultType::Rectangles);
					}
					else
					{
						continue;
					}
				}


//				UpdateConfidenceQueueMap(queueEndIndex, confidenceQueueMap, targetRects, Four);

//				UpdateVectorOfConfidenceQueueMap(confidenceQueueMap, vectorOfConfidenceQueueMap);

//				auto searchIndex = 0;

//				GetMostLiklyTargetsRect(vectorOfConfidenceQueueMap, searchIndex);

//				DrawRectangleForAllDetectedTargetsAndUpdateBlockConfidence(colorFrame, searchIndex, vectorOfConfidenceQueueMap, targetRects, confidenceValueMap, Four);


//				if (frameIndex > THINGKING_STAGE)
//				{
//					std::vector<cv::Point> blocksContainTargets;

//					UpdateConfidenceValueVector(confidenceValueMap, vectorOfConfidenceValueMap);

//					GetTopCountBlocksWhichContainsTargets(vectorOfConfidenceValueMap, blocksContainTargets);

//					for (auto i = 0; i < blocksContainTargets.size(); ++i)
//					{
//						auto findTargetFlag = false;
//						auto trackerIndex = 0;

//						auto currentBlock = blocksContainTargets[i];

//						SearchWhichTrackerForThisBlock(currentBlock, trackerIndex);

//						for (auto j = 0; j < targetRects.size(); ++j)
//						{
//							auto rect = targetRects[j];
//							auto blockXOfCurrentRect = (rect.x + rect.width / 2) / BLOCK_SIZE;
//							auto blockYOfCurrentRect = (rect.y + rect.height / 2) / BLOCK_SIZE;

//							if (blockXOfCurrentRect == currentBlock.x && blockYOfCurrentRect == currentBlock.y)
//							{
//								if (GlobalTrackerList.empty())
//								{
//									CreateNewTrackerForThisBlock(currentBlock, rect);
//								}
//								else
//								{
//									if (!UpdateTrackerStatus(rect, blockXOfCurrentRect, blockYOfCurrentRect, trackerIndex))
//									{
//										CreateNewTrackerForThisBlock(currentBlock, rect);
//									}
//								}

//								findTargetFlag = true;
//							}
//							else if ((blockXOfCurrentRect - 1 >= 0 && blockXOfCurrentRect - 1 == currentBlock.x && blockYOfCurrentRect == currentBlock.y) ||
//								(blockYOfCurrentRect - 1 >= 0 && blockXOfCurrentRect == currentBlock.x && blockYOfCurrentRect - 1 == currentBlock.y) ||
//								(blockXOfCurrentRect + 1 < countX && blockXOfCurrentRect + 1 == currentBlock.x && blockYOfCurrentRect == currentBlock.y) ||
//								(blockYOfCurrentRect + 1 < countY && blockXOfCurrentRect == currentBlock.x && blockYOfCurrentRect + 1 == currentBlock.y) ||
//								(blockYOfCurrentRect + 1 < countY && blockXOfCurrentRect + 1 < countX && blockXOfCurrentRect + 1 == currentBlock.x && blockYOfCurrentRect + 1 == currentBlock.y) ||
//								(blockYOfCurrentRect + 1 < countY && blockXOfCurrentRect - 1 >= 0 && blockXOfCurrentRect - 1 == currentBlock.x && blockYOfCurrentRect + 1 == currentBlock.y) ||
//								(blockYOfCurrentRect - 1 < countY && blockXOfCurrentRect + 1 < countX && blockXOfCurrentRect + 1 == currentBlock.x && blockYOfCurrentRect - 1 == currentBlock.y) ||
//								(blockYOfCurrentRect - 1 < countY && blockXOfCurrentRect - 1 >= 0 && blockXOfCurrentRect - 1 == currentBlock.x && blockYOfCurrentRect - 1 == currentBlock.y))
//							{
//								confidenceValueMap[currentBlock.y][currentBlock.x] = MinNeighbor(confidenceValueMap, currentBlock.y, currentBlock.x);
//								confidenceValueMap[blockYOfCurrentRect][blockXOfCurrentRect] = MaxNeighbor(confidenceValueMap, currentBlock.y, currentBlock.x);

//								if (GlobalTrackerList.empty())
//								{
//									CreateNewTrackerForThisBlock(cv::Point(blockXOfCurrentRect, blockYOfCurrentRect), rect);
//								}
//								else
//								{
//									if (!UpdateTrackerStatus(rect, blockXOfCurrentRect, blockYOfCurrentRect, trackerIndex))
//									{
//										SearchWhichTrackerForThisBlock(cv::Point(blockXOfCurrentRect, blockYOfCurrentRect), trackerIndex);
//										if (!UpdateTrackerStatus(rect, blockXOfCurrentRect, blockYOfCurrentRect, trackerIndex))
//											CreateNewTrackerForThisBlock(cv::Point(blockXOfCurrentRect, blockYOfCurrentRect), rect);
//									}
//								}
//								findTargetFlag = true;
//							}
//						}

//						if (!findTargetFlag)
//						{
//							if (trackerIndex > 0)
//							{
//								auto it = GlobalTrackerList.begin() + (trackerIndex - 1);

//								it->timeLeft--;
//								if (it->timeLeft == 0)
//								{
//									auto col = it->blockX;
//									auto row = it->blockY;

//									confidenceValueMap[row][col] /= 2;
//									if (col - 1 >= 0)
//										confidenceValueMap[row][col - 1] /= 2;
//									if (col + 1 < countX)
//										confidenceValueMap[row][col + 1] /= 2;
//									if (row - 1 >= 0)
//										confidenceValueMap[row - 1][col] /= 2;
//									if (row + 1 < countY)
//										confidenceValueMap[row + 1][col] /= 2;

//									GlobalTrackerList.erase(it);
//								}
//							}
//							else
//							{
//								auto col = currentBlock.x;
//								auto row = currentBlock.y;

//								confidenceValueMap[row][col] /= 2;
//								if (col - 1 >= 0)
//									confidenceValueMap[row][col - 1] /= 2;
//								if (col + 1 < countX)
//									confidenceValueMap[row][col + 1] /= 2;
//								if (row - 1 >= 0)
//									confidenceValueMap[row - 1][col] /= 2;
//								if (row + 1 < countY)
//									confidenceValueMap[row + 1][col] /= 2;
//							}
//						}
//					}


//					for (auto it = GlobalTrackerList.begin(); it != GlobalTrackerList.end(); ++it)
//					{
//						auto existFlag = false;
//						for (auto target : blocksContainTargets)
//						{
//							if (it->blockX == target.x && it->blockY == target.y)
//							{
//								existFlag = true;
//								break;
//							}
//						}

//						if (!existFlag)
//						{
//							it->timeLeft--;
//							if (it->timeLeft == 0)
//							{
//								auto col = it->blockX;
//								auto row = it->blockY;

//								confidenceValueMap[row][col] /= 2;
//								if (col - 1 >= 0)
//									confidenceValueMap[row][col - 1] /= 2;
//								if (col + 1 < countX)
//									confidenceValueMap[row][col + 1] /= 2;
//								if (row - 1 >= 0)
//									confidenceValueMap[row - 1][col] /= 2;
//								if (row + 1 < countY)
//									confidenceValueMap[row + 1][col] /= 2;

//								it = GlobalTrackerList.erase(it);
//								if (it == GlobalTrackerList.end())
//									break;
//							}
//						}
//					}

//					DrawResults(colorFrame);
//				}

//				ConfidenceMapUtil::LostMemory(QUEUE_SIZE, queueEndIndex, confidenceQueueMap);

				imshow("Last Detect and Tracking Result", colorFrame);

				WriteLastResultToDisk(colorFrame, frameIndex, writeFileName);

				std::cout << "Index : " << std::setw(4) << frameIndex++ << std::endl;
			}
		}

		cv::destroyAllWindows();
	}

	else
	{
		std::cout << "Open Image List Failed" << std::endl;
	}

	system("pause");
	return 0;
}
