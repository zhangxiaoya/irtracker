#pragma once
#include <core/core.hpp>
#include "Util.hpp"
#include <highgui/highgui.hpp>
#include <iostream>

class DetectByMaxFilterAndAdptiveThreshHold
{
public:

	static void MergeCrossedRectangles(const std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects);

	static void Detect(cv::Mat curFrame);

private:

	static void MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize);

	static unsigned char GetMaxPixelValue(const cv::Mat& curFrame, std::vector<uchar>& pixelValues, int r, int c, int kernelSize);

	static int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap);

	static bool GetTopValues(const cv::Mat filtedFrame, std::vector<uchar>& maxValues, int topCount);

};

inline bool DetectByMaxFilterAndAdptiveThreshHold::GetTopValues(const cv::Mat filtedFrame, std::vector<uchar>& maxValues, int topCount)
{
	std::vector<uchar> allValues;

	for (auto r = 0; r < filtedFrame.rows; ++r)
		for (auto c = 0; c < filtedFrame.cols; ++c)
			allValues.push_back(filtedFrame.at<uchar>(r, c));

	sort(allValues.begin(), allValues.end(), Util::comp);

	auto iterator = unique(allValues.begin(), allValues.end());
	allValues.resize(distance(allValues.begin(), iterator));
	for (auto i = 0; i < topCount; ++i)
		maxValues[i] = allValues[i];

	if (allValues.size() < topCount)
		return false;

	return true;
}

inline void DetectByMaxFilterAndAdptiveThreshHold::MergeCrossedRectangles(const std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects)
{

}

inline void DetectByMaxFilterAndAdptiveThreshHold::Detect(cv::Mat curFrame)
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

	const auto topCount = 5;
	std::vector<uchar> maxValues(topCount, 0);

	if(!GetTopValues(filtedFrame, maxValues, topCount))
		return;

	cv::Mat blockMap(cv::Size(filtedFrame.cols, filtedFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks(filtedFrame, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects);

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(allObjects,afterMergeObjects);

	std::cout << "Max Value Threh Hold = " << static_cast<int>(maxValues[4]) << std::endl;
	Util::ShowAllObject(curFrame, allObjects);
	Util::ShowCandidateTargets(curFrame, allObjects, maxValues[4]);
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
