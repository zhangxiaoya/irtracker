#pragma once
#include <core/core.hpp>
#include "Util.hpp"
#include <highgui/highgui.hpp>
#include <iostream>
#include <filesystem>

class DetectByMaxFilterAndAdptiveThreshHold
{
public:

	static std::vector<cv::Rect> Detect(cv::Mat curFrame);

private:

	static void MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize);

	static unsigned char GetMaxPixelValue(const cv::Mat& curFrame, std::vector<uchar>& pixelValues, int r, int c, int kernelSize);

	static int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap);

	static void MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects);

	static bool GetTopValues(const cv::Mat filtedFrame, uchar& pixelThreshHold, int topCount);

	static bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond);

	static void RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects, const cv::Mat& frame, uchar threshHold);

};

inline bool DetectByMaxFilterAndAdptiveThreshHold::GetTopValues(const cv::Mat filtedFrame, uchar& pixelThreshHold, int topCount)
{
	std::vector<uchar> allValues;

	for (auto r = 0; r < filtedFrame.rows; ++r)
		for (auto c = 0; c < filtedFrame.cols; ++c)
			allValues.push_back(filtedFrame.at<uchar>(r, c));

	sort(allValues.begin(), allValues.end(), Util::comp);

	auto iterator = unique(allValues.begin(), allValues.end());
	allValues.resize(distance(allValues.begin(), iterator));

	if (allValues.size() < topCount)
		return false;

	pixelThreshHold = allValues[topCount - 1];
	return true;
}

inline bool DetectByMaxFilterAndAdptiveThreshHold::CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond)
{
	auto firstCenterX = (objectFirst.right + objectFirst.left) / 2;
	auto firstCenterY = (objectFirst.bottom + objectFirst.top) / 2;

	auto secondCenterX = (objectSecond.right + objectSecond.left) / 2;
	auto secondCenterY = (objectSecond.bottom + objectSecond.top) / 2;

	auto firstWidth = objectFirst.right - objectFirst.left + 1;
	auto firstHeight = objectFirst.bottom - objectFirst.top + 1;

	auto secondWidth = objectSecond.right - objectSecond.left + 1;
	auto secondHeight = objectSecond.bottom - objectSecond.top + 1;

	auto centerXDiff = std::abs(firstCenterX - secondCenterX);
	auto centerYDiff = std::abs(firstCenterY - secondCenterY);

	if (centerXDiff <= (firstWidth + secondWidth) / 2 && centerYDiff <= (firstHeight + secondHeight) / 2)
		return true;

	return false;
}

inline void DetectByMaxFilterAndAdptiveThreshHold::RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects, const cv::Mat& frame, uchar threshHold)
{
	for (auto it = allObjects.begin(); it != allObjects.end();)
	{
		auto width = it->right - it->left + 1;
		auto height = it->bottom - it->top + 1;
		if (width < 3 || height < 3 || width > 10 || height > 10 || frame.at<uchar>(it->top+1, it->left+1) < threshHold)
			it = allObjects.erase(it);
		else
			++it;
	}
}

inline void DetectByMaxFilterAndAdptiveThreshHold::MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects)
{
	for (auto i = 0; i < allObjects.size() - 1; ++i)
	{
		for (auto j = i + 1; j < allObjects.size();++j)
		{
			if (i == 19)
				auto dummy = 0;
			if(allObjects[j].identify == allObjects[i].identify)
				continue;
			if(CheckCross(allObjects[i], allObjects[j]))
			{
				allObjects[j].identify = allObjects[i].identify;
			}
		}
	}

	while(true)
	{
		auto it = allObjects.begin();
		while(it != allObjects.end() && it->identify == -1)
			++it;

		if(it == allObjects.end())
			break;

		afterMergeObjects.push_back(*it);

		it->identify = -1;
		++it;

		while(it != allObjects.end())
		{
			if(it->identify != -1 && it->identify == afterMergeObjects[afterMergeObjects.size()-1].identify)
			{
				if (it->top < afterMergeObjects[afterMergeObjects.size() - 1].top)
					afterMergeObjects[afterMergeObjects.size() - 1].top = it->top;

				if (it->left < afterMergeObjects[afterMergeObjects.size() - 1].left)
					afterMergeObjects[afterMergeObjects.size() - 1].left = it->left;

				if (it->right > afterMergeObjects[afterMergeObjects.size() - 1].right)
					afterMergeObjects[afterMergeObjects.size() - 1].right = it->right;

				if (it->bottom > afterMergeObjects[afterMergeObjects.size() - 1].bottom)
					afterMergeObjects[afterMergeObjects.size() - 1].bottom = it->bottom;

				it->identify = -1;
			}
			++it;
		}
	}
}

inline std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshHold::Detect(cv::Mat curFrame)
{
 	cv::Mat filtedFrame(cv::Size(curFrame.cols, curFrame.rows), CV_8UC1);
	auto kernelSize = 3;

	if(AFTER_MAX_FILTER)
	{
		curFrame.copyTo(filtedFrame);
	}
	else
	{
		MaxFilter(curFrame, filtedFrame, kernelSize);
	}

	imshow("Max Filter", filtedFrame);

	const auto topCount = 8;
	uchar pixelThreshHold = 0;

	cv::Mat blockMap(cv::Size(filtedFrame.cols, filtedFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks(filtedFrame, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects);

	std::vector<cv::Rect> falseResult;
	if (!GetTopValues(filtedFrame, pixelThreshHold, topCount))
		return falseResult;

	RemoveSmallAndBigObjects(allObjects,filtedFrame, pixelThreshHold);

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(allObjects,afterMergeObjects);

	std::cout << "Max Value Threh Hold = " << static_cast<int>(pixelThreshHold) << std::endl;
	Util::ShowAllObject(curFrame, afterMergeObjects);
	Util::ShowCandidateTargets(curFrame, afterMergeObjects, pixelThreshHold);

	return Util::GetCandidateTargets(curFrame, afterMergeObjects, pixelThreshHold);
}

inline void DetectByMaxFilterAndAdptiveThreshHold::MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize)
{
	std::vector<uchar> pixelVector;

	for (auto r = 0; r < curFrame.rows; ++r)
	{
		for (auto c = 0; c < curFrame.cols; ++c)
		{
			pixelVector.clear();
			filtedFrame.at<uchar>(r, c) = GetMaxPixelValue(curFrame, pixelVector, r, c, kernelSize);
		}
	}
}

inline unsigned char DetectByMaxFilterAndAdptiveThreshHold::GetMaxPixelValue(const cv::Mat& curFrame, std::vector<uchar>& pixelValues, int r, int c, int kernelSize)
{
	auto radius = kernelSize / 2;
	auto leftTopX = c - radius;
	auto leftTopY = r - radius;

	auto rightBottomX = leftTopX + 2 * radius;
	auto rightBottomY = leftTopY + 2 * radius;

	for (auto row = leftTopY; row <= rightBottomY; ++row)
	{
		if (row >= 0 && row < curFrame.rows)
		{
			for (auto col = leftTopX; col <= rightBottomX; ++col)
			{
				if (col >= 0 && col < curFrame.cols)
					pixelValues.push_back(curFrame.at<uchar>(row, col));
			}
		}
	}

	return Util::MaxOfVector(pixelValues.begin(), pixelValues.end());
}

inline int DetectByMaxFilterAndAdptiveThreshHold::GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap)
{
	auto currentIndex = 0;
	for (auto r = 0; r < filtedFrame.rows; ++r)
	{
		for (auto c = 0; c < filtedFrame.cols; ++c)
		{
			if (blockMap.at<int32_t>(r, c) != -1)
				continue;

			auto val = filtedFrame.at<uchar>(r, c);
			Util::FindNeighbor(filtedFrame, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	return currentIndex;
}
