#pragma once
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <filesystem>
#include "../Models/FourLimits.hpp"
#include "../Utils/Util.hpp"
#include "../DifferenceElem.hpp"

cv::Mat previousFrame = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_32SC1, cv::Scalar(1));

class DetectByMaxFilterAndAdptiveThreshold
{
public:

	template <typename DATA_TYPE>
	static std::vector<cv::Rect> Detect(cv::Mat& curFrame, cv::Mat& fdImg);

private:

	template <typename DATA_TYPE>
	static void MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize);

	template <typename DATA_TYPE>
	static DATA_TYPE GetMaxPixelValue(const cv::Mat& curFrame, int r, int c, int kernelSize);

	template <typename DATA_TYPE>
	static int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap);

	template <typename DATA_TYPE>
	static void Discretization(const cv::Mat& filtedFrame, cv::Mat& discretizatedFrame);

	static void MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects);

	static void RefreshMask(cv::Mat curFrame, std::vector<cv::Rect> result);

	static void FilterRectByContinuty(cv::Mat curFrame, std::vector<cv::Rect> rects, std::vector<cv::Rect> result);

	template<typename DATA_TYPE>
	static std::vector<std::vector<DATA_TYPE>> GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame);

	template<typename DATA_TYPE>
	static void StrengthenIntensityOfBlock(cv::Mat& curFrame);

	template <typename DATA_TYPE>
	static void GetMaxValueOfMatrix(std::vector<std::vector<DATA_TYPE>> maxmindiff, DifferenceElem& diffElem);

	template <typename DATA_TYPE>
	static std::vector<DifferenceElem> GetMostMaxDiffBlock(std::vector<std::vector<DATA_TYPE>> maxmindiff);

	static bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond);

	template <typename DATA_TYPE>
	static void CalculateThreshold(const cv::Mat& frame, DATA_TYPE& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

	template <typename DATA_TYPE>
	static void RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects, const cv::Mat& frame);

	static void FillRectToFrame(cv::Rect& rect);

	static bool CheckRect(cv::Rect& rect);
};

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

template <typename DATA_TYPE>
void DetectByMaxFilterAndAdptiveThreshold::CalculateThreshold(const cv::Mat& frame, DATA_TYPE& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	auto sumAll = 0;
	for (auto r = leftTopY; r < rightBottomY; ++ r)
	{
		auto sumRow = 0;
		for (auto c = leftTopX; c < rightBottomX; ++c)
		{
			sumRow += frame.at<DATA_TYPE>(r, c);
		}
		sumAll += (sumRow / (rightBottomX - leftTopX));
	}

	threshHold = sumAll / (rightBottomY - leftTopY);

	//	threshHold += (threshHold) / 4;
}

template <typename DATA_TYPE>
void DetectByMaxFilterAndAdptiveThreshold::RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects, const cv::Mat& frame)
{
	for (auto it = allObjects.begin(); it != allObjects.end();)
	{
		DATA_TYPE threshold = 0;

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

		CalculateThreshold<DATA_TYPE>(frame, threshold, leftTopX, leftTopY, rightBottomX, rightBottomY);

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT) ||
			frame.at<DATA_TYPE>(it->top + 1, it->left + 1) < threshold)
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

template<typename DATA_TYPE>
std::vector<std::vector<DATA_TYPE>> DetectByMaxFilterAndAdptiveThreshold::GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame)
{
	std::vector<std::vector<DATA_TYPE>> maxmindiff(countY, std::vector<DATA_TYPE>(countX, 0));
	for (auto br = 0; br < countY; ++br)
	{
		auto height = br == (countY - 1) ? IMAGE_HEIGHT - (countY - 1) * BLOCK_SIZE : BLOCK_SIZE;
		for (auto bc = 0; bc < countX; ++bc)
		{
			auto width = bc == (countX - 1) ? IMAGE_WIDTH - (countX - 1) * BLOCK_SIZE : BLOCK_SIZE;
			maxmindiff[br][bc] =
				Util::GetMaxValueOfBlock<DATA_TYPE>(curFrame(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height))) -
				Util::GetMinValueOfBlock<DATA_TYPE>(curFrame(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height)));
		}
	}
	return maxmindiff;
}

template<typename DATA_TYPE>
void DetectByMaxFilterAndAdptiveThreshold::StrengthenIntensityOfBlock(cv::Mat& curFrame)
{
	auto maxmindiffMatrix = GetMaxMinPixelValueDifferenceMap<DATA_TYPE>(curFrame);
	auto differenceElems = GetMostMaxDiffBlock<DATA_TYPE>(maxmindiffMatrix);

	for (auto elem : differenceElems)
	{
		auto centerX = elem.blockX * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto centerY = elem.blockY * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto boundingBoxLeftTopX = centerX - BLOCK_SIZE >= 0 ? centerX - BLOCK_SIZE : 0;
		auto boundingBoxLeftTopY = centerY - BLOCK_SIZE >= 0 ? centerY - BLOCK_SIZE : 0;
		auto boundingBoxRightBottomX = centerX + BLOCK_SIZE < IMAGE_WIDTH ? centerX + BLOCK_SIZE : IMAGE_WIDTH - 1;
		auto boundingBoxRightBottomY = centerY + BLOCK_SIZE < IMAGE_HEIGHT ? centerY + BLOCK_SIZE : IMAGE_HEIGHT - 1;
		auto averageValue = Util::CalculateAverageValue<DATA_TYPE>(curFrame, boundingBoxLeftTopX, boundingBoxLeftTopY, boundingBoxRightBottomX, boundingBoxRightBottomY);

		auto maxdiffBlockRightBottomX = (elem.blockX + 1) * BLOCK_SIZE > IMAGE_WIDTH ? IMAGE_WIDTH - 1 : (elem.blockX + 1) * BLOCK_SIZE;
		auto maxdiffBlockRightBottomY = (elem.blockY + 1) * BLOCK_SIZE > IMAGE_HEIGHT ? IMAGE_HEIGHT - 1 : (elem.blockY + 1) * BLOCK_SIZE;
		for (auto r = elem.blockY * BLOCK_SIZE; r < maxdiffBlockRightBottomY; ++r)
		{
			for (auto c = elem.blockX * BLOCK_SIZE; c < maxdiffBlockRightBottomX; ++c)
			{
				if (curFrame.at<DATA_TYPE>(r, c) > averageValue)
				{
					curFrame.at<DATA_TYPE>(r, c) = curFrame.at<DATA_TYPE>(r, c) + 10 > 255 ? 255 : curFrame.at<DATA_TYPE>(r, c) + 10;
				}
			}
		}
	}
}

template <typename DATA_TYPE>
void DetectByMaxFilterAndAdptiveThreshold::GetMaxValueOfMatrix(std::vector<std::vector<DATA_TYPE>> maxmindiff, DifferenceElem& diffElem)
{
	for (auto br = 0; br < countY; ++br)
	{
		for (auto bc = 0; bc < countX; ++bc)
		{
			if(diffElem.diffVal < static_cast<int>(maxmindiff[br][bc]))
			{
				diffElem.diffVal = static_cast<int>(maxmindiff[br][bc]);
				diffElem.blockX = bc;
				diffElem.blockY = br;
			}
		}
	}
}

template <typename DATA_TYPE>
std::vector<DifferenceElem> DetectByMaxFilterAndAdptiveThreshold::GetMostMaxDiffBlock(std::vector<std::vector<DATA_TYPE>> maxmindiff)
{
	std::vector<DifferenceElem> mostPossibleBlocks;

	DifferenceElem diffElem;
	GetMaxValueOfMatrix<DATA_TYPE>(maxmindiff, diffElem);
	mostPossibleBlocks.push_back(diffElem);

	return mostPossibleBlocks;
}

template <typename DATA_TYPE>
std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold::Detect(cv::Mat& curFrame, cv::Mat& fdImg)
{
	cv::Mat filtedFrame(cv::Size(curFrame.cols, curFrame.rows), CV_DATA_TYPE);
	auto kernelSize = 3;

	StrengthenIntensityOfBlock<DATA_TYPE>(curFrame);

	MaxFilter<DATA_TYPE>(curFrame, filtedFrame, kernelSize);

	cv::Mat discrezatedFrame(cv::Size(curFrame.cols, curFrame.rows), CV_DATA_TYPE);

	Discretization<DATA_TYPE>(filtedFrame, discrezatedFrame);

	fdImg = discrezatedFrame;

	imshow("Max Filter and Discrezated", discrezatedFrame);

	cv::Mat blockMap(cv::Size(discrezatedFrame.cols, discrezatedFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks<DATA_TYPE>(discrezatedFrame, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects);

	Util::ShowAllObject(curFrame, allObjects, "Before Merge and Remove out scale Objects");

	RemoveSmallAndBigObjects<DATA_TYPE>(allObjects, discrezatedFrame);

	Util::ShowAllObject(curFrame, allObjects, "Before Merge");

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(allObjects, afterMergeObjects);

	Util::ShowAllObject(curFrame, afterMergeObjects, "After Merge Cross Rectangles");

	auto rects = Util::GetCandidateTargets<DATA_TYPE>(discrezatedFrame, afterMergeObjects);

	Util::ShowAllCandidateTargets(curFrame, rects);
	std::cout << "Count = " << rects.size() << std::endl;

	return rects;
}

template <typename DATA_TYPE>
void DetectByMaxFilterAndAdptiveThreshold::MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize)
{
	for (auto r = 0; r < curFrame.rows; ++r)
	{
		for (auto c = 0; c < curFrame.cols; ++c)
		{
			filtedFrame.at<DATA_TYPE>(r, c) = GetMaxPixelValue<DATA_TYPE>(curFrame, r, c, kernelSize);
		}
	}
}

template <typename DATA_TYPE>
DATA_TYPE DetectByMaxFilterAndAdptiveThreshold::GetMaxPixelValue(const cv::Mat& curFrame, int r, int c, int kernelSize)
{
	auto radius = kernelSize / 2;
	auto leftTopX = c - radius;
	auto leftTopY = r - radius;

	auto rightBottomX = leftTopX + 2 * radius;
	auto rightBottomY = leftTopY + 2 * radius;

	DATA_TYPE maxVal = 0;

	for (auto row = leftTopY; row <= rightBottomY; ++row)
	{
		if (row >= 0 && row < curFrame.rows)
		{
			for (auto col = leftTopX; col <= rightBottomX; ++col)
			{
				if (col >= 0 && col < curFrame.cols && maxVal < curFrame.at<DATA_TYPE>(row, col))
					maxVal = curFrame.at<DATA_TYPE>(row, col);
			}
		}
	}

	return maxVal;
}

template <typename DATA_TYPE>
int DetectByMaxFilterAndAdptiveThreshold::GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap)
{
	auto currentIndex = 0;
	for (auto r = 0; r < filtedFrame.rows; ++r)
	{
		for (auto c = 0; c < filtedFrame.cols; ++c)
		{
			if (blockMap.at<int32_t>(r, c) != -1)
				continue;

			auto val = filtedFrame.at<DATA_TYPE>(r, c);
			Util::FindNeighbor<DATA_TYPE>(filtedFrame, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	return currentIndex;
}

template <typename DATA_TYPE>
void DetectByMaxFilterAndAdptiveThreshold::Discretization(const cv::Mat& filtedFrame, cv::Mat& discretizatedFrame)
{
	for (auto r = 0; r < filtedFrame.rows; ++r)
		for (auto c = 0; c < filtedFrame.cols; ++c)
			discretizatedFrame.at<DATA_TYPE>(r, c) = (filtedFrame.at<DATA_TYPE>(r, c) / DISCRATED_BIN) * DISCRATED_BIN;
}

