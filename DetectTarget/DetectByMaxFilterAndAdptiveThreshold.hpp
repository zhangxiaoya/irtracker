#pragma once
#include <core/core.hpp>
#include "Util.hpp"
#include <highgui/highgui.hpp>
#include <iostream>
#include <filesystem>

cv::Mat previousFrame = cv::Mat(cv::Size(320, 256), CV_32SC1, cv::Scalar(1));

class DetectByMaxFilterAndAdptiveThreshold
{
public:

	static std::vector<cv::Rect> Detect(cv::Mat curFrame);

private:

	static void MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize);

	static unsigned char GetMaxPixelValue(const cv::Mat& curFrame, std::vector<uchar>& pixelValues, int r, int c, int kernelSize);

	static int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap);

	static void Discretization(const cv::Mat& filtedFrame, cv::Mat& discretizatedFrame, uint8_t bin);

	static void MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects);

	static void RefreshMask(cv::Mat curFrame, std::vector<cv::Rect> result);

	static void FilterRectByContinuty(cv::Mat curFrame, std::vector<cv::Rect> rects, std::vector<cv::Rect> result);

	static bool GetTopValues(const cv::Mat filtedFrame, uchar& pixelThreshHold, int topCount);

	static bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond);

	static void CalculateThreshold(const cv::Mat& frame, uchar& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

	static void RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects, const cv::Mat& frame);

	static void FillRectToFrame(cv::Rect& rect);

	static bool CheckRect(cv::Rect& rect);
};

inline bool DetectByMaxFilterAndAdptiveThreshold::GetTopValues(const cv::Mat filtedFrame, uchar& pixelThreshHold, int topCount)
{
	std::vector<uchar> allValues;

	for (auto r = 0; r < filtedFrame.rows; ++r)
		for (auto c = 0; c < filtedFrame.cols; ++c)
			allValues.push_back(filtedFrame.at<uchar>(r, c));

	sort(allValues.begin(), allValues.end(), Util::CompareUchar);

	auto iterator = unique(allValues.begin(), allValues.end());
	allValues.resize(distance(allValues.begin(), iterator));

	if (allValues.size() < topCount)
		return false;

	pixelThreshHold = allValues[topCount - 1];
	return true;
}

inline bool DetectByMaxFilterAndAdptiveThreshold::CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond)
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

inline void DetectByMaxFilterAndAdptiveThreshold::CalculateThreshold(const cv::Mat& frame, uchar& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	auto sumAll = 0;
	for (auto r = leftTopY; r < rightBottomY; ++ r)
	{
		auto sumRow = 0;
		for (auto c = leftTopX; c < rightBottomX; ++c)
		{
			sumRow += frame.at<uchar>(r, c);
		}
		sumAll += (sumRow / (rightBottomX - leftTopX));
	}

	threshHold = sumAll / (rightBottomY - leftTopY);

	threshHold += (threshHold) / 4;
}

inline void DetectByMaxFilterAndAdptiveThreshold::RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects, const cv::Mat& frame)
{
	for (auto it = allObjects.begin(); it != allObjects.end();)
	{
		uchar threshHold = 0;

		auto width = it->right - it->left + 1;
		auto height = it->bottom - it->top + 1;

		auto surroundBoxWidth = 2 * width;
		auto surroundBoxHeight = 2 * height;

		auto centerX = (it->right + it->left) / 2;
		auto centerY = (it->bottom + it->top) / 2;

		auto leftTopX = centerX - surroundBoxWidth / 2;
		if (leftTopX < 0)
			leftTopX = 0;

		auto leftTopY = centerY - surroundBoxHeight / 2;
		if (leftTopY < 0)
			leftTopY = 0;

		auto rightBottomX = leftTopX + surroundBoxWidth;
		if (rightBottomX > frame.cols)
			rightBottomX = frame.cols;

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY > frame.rows)
			rightBottomY = frame.rows;

		CalculateThreshold(frame, threshHold, leftTopX, leftTopY, rightBottomX, rightBottomY);

		if (width < 3 || height < 3 || width > 10 || height > 10 || frame.at<uchar>(it->top + 1, it->left + 1) < threshHold)
			it = allObjects.erase(it);
		else
			++it;
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::FillRectToFrame(cv::Rect& rect)
{
	for (auto r = rect.y; r < rect.y + rect.height; ++r)
	{
		for (auto c = rect.x; c < rect.x + rect.width; ++c)
		{
			previousFrame.at<int32_t>(r, c) = 1;
		}
	}
}

inline bool DetectByMaxFilterAndAdptiveThreshold::CheckRect(cv::Rect& rect)
{
	auto leftTopX = rect.x - rect.width > 0 ? rect.x - rect.width > 0 : 0;
	auto leftTopY = rect.y - rect.height > 0 ? rect.y - rect.height > 0 : 0;
	auto rightBottomX = rect.x + 2 * rect.width < previousFrame.cols ? rect.x + 2 * rect.width : previousFrame.cols - 1;
	auto rightBottomY = rect.y + 2 * rect.height < previousFrame.rows ? rect.y + 2 * rect.height : previousFrame.rows - 1;

	auto count = 0;
	for (auto r = leftTopY; r <= rightBottomY; ++r)
	{
		for (auto c = leftTopX; c <= rightBottomX; ++c)
		{
			if (previousFrame.at<int32_t>(r, c) == 1)
				++count;
		}
	}
	auto x = static_cast<double>(count) / (rect.width * rect.height * 4);
	if (x > 0.2)
		return true;
	return false;
}

inline void DetectByMaxFilterAndAdptiveThreshold::MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects)
{
	for (auto i = 0; i < allObjects.size(); ++i)
	{
		if (allObjects[i].identify == -1)
			continue;
		for (auto j = 0; j < allObjects.size(); ++j)
		{
			if (i == j || allObjects[j].identify == -1)
				continue;
			if (CheckCross(allObjects[i], allObjects[j]))
			{
				allObjects[j].identify = -1;

				if (allObjects[i].top > allObjects[j].top)
					allObjects[i].top = allObjects[j].top;

				if (allObjects[i].left > allObjects[j].left)
					allObjects[i].left = allObjects[j].left;

				if (allObjects[i].right < allObjects[j].right)
					allObjects[i].right = allObjects[j].right;

				if (allObjects[i].bottom < allObjects[j].bottom)
					allObjects[i].bottom = allObjects[j].bottom;
			}
		}
	}
	// for left top may be missed, so need double check
	for (auto i = 0; i < allObjects.size(); ++i)
	{
		if (allObjects[i].identify == -1)
			continue;
		for (auto j = 0; j < allObjects.size(); ++j)
		{
			if (i == j || allObjects[j].identify == -1)
				continue;
			if (CheckCross(allObjects[i], allObjects[j]))
			{
				allObjects[j].identify = -1;

				if (allObjects[i].top > allObjects[j].top)
					allObjects[i].top = allObjects[j].top;

				if (allObjects[i].left > allObjects[j].left)
					allObjects[i].left = allObjects[j].left;

				if (allObjects[i].right < allObjects[j].right)
					allObjects[i].right = allObjects[j].right;

				if (allObjects[i].bottom < allObjects[j].bottom)
					allObjects[i].bottom = allObjects[j].bottom;
			}
		}
	}

	for (auto i = 0; i < allObjects.size(); ++i)
	{
		if (allObjects[i].identify != -1)
			afterMergeObjects.push_back(allObjects[i]);
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::RefreshMask(cv::Mat curFrame, std::vector<cv::Rect> result)
{
	previousFrame.release();
	previousFrame = cv::Mat(cv::Size(curFrame.cols, curFrame.rows), CV_32SC1, cv::Scalar(-1));
	for (auto i = 0; i < result.size(); ++i)
		FillRectToFrame(result[i]);
}

inline void DetectByMaxFilterAndAdptiveThreshold::FilterRectByContinuty(cv::Mat curFrame, std::vector<cv::Rect> rects, std::vector<cv::Rect> result)
{
	for (auto it = rects.begin(); it != rects.end(); ++it)
	{
		if (CheckRect(*it))
			result.push_back(*it);
	}

	if (result.size() >= 2)
		RefreshMask(curFrame, result);
	else
		RefreshMask(curFrame, rects);
}

inline std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold::Detect(cv::Mat curFrame)
{
	cv::Mat filtedFrame(cv::Size(curFrame.cols, curFrame.rows), CV_8UC1);
	auto kernelSize = 3;

	MaxFilter(curFrame, filtedFrame, kernelSize);

	cv::Mat discrezatedFrame(cv::Size(curFrame.cols, curFrame.rows), CV_8UC1);
	auto bin = 15;

	Discretization(filtedFrame, discrezatedFrame, bin);

	imshow("Max Filter and Discrezated", discrezatedFrame);


	cv::Mat blockMap(cv::Size(discrezatedFrame.cols, discrezatedFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks(discrezatedFrame, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects);


	RemoveSmallAndBigObjects(allObjects, discrezatedFrame);

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(allObjects, afterMergeObjects);

	Util::ShowAllObject(curFrame, afterMergeObjects);

	auto rects = Util::GetCandidateTargets(curFrame, afterMergeObjects);

	Util::ShowAllCandidateTargets(curFrame, rects);

	return rects;
}

inline void DetectByMaxFilterAndAdptiveThreshold::MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize)
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

inline unsigned char DetectByMaxFilterAndAdptiveThreshold::GetMaxPixelValue(const cv::Mat& curFrame, std::vector<uchar>& pixelValues, int r, int c, int kernelSize)
{
	auto radius = kernelSize / 2;
	auto leftTopX = c - radius;
	auto leftTopY = r - radius;

	auto rightBottomX = leftTopX + 2 * radius;
	auto rightBottomY = leftTopY + 2 * radius;

	uchar maxVal = 0;

	for (auto row = leftTopY; row <= rightBottomY; ++row)
	{
		if (row >= 0 && row < curFrame.rows)
		{
			for (auto col = leftTopX; col <= rightBottomX; ++col)
			{
				if (col >= 0 && col < curFrame.cols && maxVal < curFrame.at<uchar>(row, col))
					maxVal = curFrame.at<uchar>(row, col);
			}
		}
	}

	return maxVal;
}

inline int DetectByMaxFilterAndAdptiveThreshold::GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap)
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

inline void DetectByMaxFilterAndAdptiveThreshold::Discretization(const cv::Mat& filtedFrame, cv::Mat& discretizatedFrame, uint8_t bin)
{
	for (auto r = 0; r < filtedFrame.rows; ++r)
		for (auto c = 0; c < filtedFrame.cols; ++c)
			discretizatedFrame.at<uint8_t>(r, c) = (filtedFrame.at<uint8_t>(r, c) / bin) * bin;
}
