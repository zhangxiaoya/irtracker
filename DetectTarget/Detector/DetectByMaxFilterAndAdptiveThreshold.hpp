#pragma once
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include "../Models/FourLimits.hpp"
#include "../Utils/Util.hpp"
#include "../DifferenceElem.hpp"
#include "../Utils/PerformanceUtil.hpp"

cv::Mat previousFrame = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_32SC1, cv::Scalar(1));

class DetectByMaxFilterAndAdptiveThreshold
{
public:

	template<typename DataType>
	static std::vector<cv::Rect> Detect(cv::Mat& curFrame, cv::Mat& preprocessResultFrame, cv::Mat& detectedResultFrame);

private:

	static void MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize);

	static int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap);

	static void Discretization(const cv::Mat& filtedFrame, cv::Mat& discretizatedFrame);

	static void MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects);

	static void RefreshMask(cv::Mat curFrame, std::vector<cv::Rect> result);

	static void FilterRectByContinuty(cv::Mat curFrame, std::vector<cv::Rect> rects, std::vector<cv::Rect> result);

	static std::vector<std::vector<uchar>> GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame);

	static void StrengthenIntensityOfBlock(cv::Mat& curFrame);

	static void GetMaxValueOfMatrix(std::vector<std::vector<uchar>> maxmindiff, DifferenceElem& diffElem);

	static std::vector<DifferenceElem> GetMostMaxDiffBlock(std::vector<std::vector<uchar>> maxmindiff);

	static void SearchNeighbors(const std::vector<std::vector<unsigned char>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal);

	static void GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<uchar>> maxmindiff, std::vector<DifferenceElem>& diffElemVec);

	static bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond);

	static void CalculateThreshold(const cv::Mat& frame, uchar& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

	static void RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects);

	static void RemoveObjectsWithLowContrast(std::vector<FourLimits>& allObjects, const cv::Mat& frame);

	static void DoubleCheckAfterMerge(const cv::Mat& frame, std::vector<FourLimits>& allObjects);

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

	if (centerXDiff <= (firstWidth + secondWidth) / 2 + 1 && centerYDiff <= (firstHeight + secondHeight) / 2 + 1)
		return true;

	return false;
}

inline void DetectByMaxFilterAndAdptiveThreshold::CalculateThreshold(const cv::Mat& frame, uchar& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	auto sumAll = 0;
	for (auto r = leftTopY; r < rightBottomY; ++ r)
	{
		auto sumRow = 0;
		auto ptr = frame.ptr<uchar>(r);
		for (auto c = leftTopX; c < rightBottomX; ++c)
		{
			sumRow += static_cast<int>(ptr[c]);
		}
		sumAll += static_cast<int>(sumRow / (rightBottomX - leftTopX));
	}

	threshHold = static_cast<uchar>(sumAll / (rightBottomY - leftTopY));

	//	threshHold += (threshHold) / 4;
}

inline void DetectByMaxFilterAndAdptiveThreshold::RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects)
{
	for (auto it = allObjects.begin(); it != allObjects.end();)
	{
		auto width = it->right - it->left + 1;
		auto height = it->bottom - it->top + 1;

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			it = allObjects.erase(it);
		else
			++it;
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::RemoveObjectsWithLowContrast(std::vector<FourLimits>& allObjects, const cv::Mat& frame)
{
	for (auto it = allObjects.begin(); it != allObjects.end();)
	{
		uchar threshold = 0;
		uchar centerValue = 0;

		auto width = it->right - it->left + 1;
		auto height = it->bottom - it->top + 1;

		auto surroundBoxWidth = 3 * width;
		auto surroundBoxHeight = 3 * height;

		auto centerX = (it->right + it->left) / 2;
		auto centerY = (it->bottom + it->top) / 2;

		auto leftTopX = centerX - surroundBoxWidth / 2;
		if (leftTopX < 0)
		{
			leftTopX = 0;
		}

		auto leftTopY = centerY - surroundBoxHeight / 2;
		if (leftTopY < 0)
		{
			leftTopY = 0;
		}

		auto rightBottomX = leftTopX + surroundBoxWidth;
		if (rightBottomX > frame.cols)
		{
			rightBottomX = frame.cols;
		}

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY > frame.rows)
		{
			rightBottomY = frame.rows;
		}

		Util::CalculateThreshHold(frame, threshold, leftTopX, leftTopY, rightBottomX, rightBottomY);

		Util::CalCulateCenterValue(frame, centerValue, cv::Rect(it->left, it->top, it->right - it->left + 1, it->bottom - it->top + 1));

		if (std::abs(static_cast<int>(centerValue) - static_cast<int>(threshold)) < 3)
		{
			it = allObjects.erase(it);
		}
		else
		{
			++it;
		}
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::DoubleCheckAfterMerge(const cv::Mat& frame, std::vector<FourLimits>& allObjects)
{
	RemoveSmallAndBigObjects(allObjects);
	RemoveObjectsWithLowContrast(allObjects, frame);
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

inline std::vector<std::vector<uchar>> DetectByMaxFilterAndAdptiveThreshold::GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame)
{
	std::vector<std::vector<uchar>> maxmindiff(countY, std::vector<uchar>(countX, 0));
	for (auto br = 0; br < countY; ++br)
	{
		auto height = br == (countY - 1) ? IMAGE_HEIGHT - (countY - 1) * BLOCK_SIZE : BLOCK_SIZE;
		for (auto bc = 0; bc < countX; ++bc)
		{
			auto width = bc == (countX - 1) ? IMAGE_WIDTH - (countX - 1) * BLOCK_SIZE : BLOCK_SIZE;
			maxmindiff[br][bc] = Util::GetMaxValueOfBlock(curFrame(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height))) - Util::GetMinValueOfBlock(curFrame(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height)));
		}
	}
	return maxmindiff;
}

inline void DetectByMaxFilterAndAdptiveThreshold::StrengthenIntensityOfBlock(cv::Mat& currentGrayFrame)
{
	auto maxmindiffMatrix = GetMaxMinPixelValueDifferenceMap(currentGrayFrame);

	auto differenceElems = GetMostMaxDiffBlock(maxmindiffMatrix);

	for (auto elem : differenceElems)
	{
		auto centerX = elem.blockX * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto centerY = elem.blockY * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto boundingBoxLeftTopX = centerX - BLOCK_SIZE >= 0 ? centerX - BLOCK_SIZE : 0;
		auto boundingBoxLeftTopY = centerY - BLOCK_SIZE >= 0 ? centerY - BLOCK_SIZE : 0;
		auto boundingBoxRightBottomX = centerX + BLOCK_SIZE < IMAGE_WIDTH ? centerX + BLOCK_SIZE : IMAGE_WIDTH - 1;
		auto boundingBoxRightBottomY = centerY + BLOCK_SIZE < IMAGE_HEIGHT ? centerY + BLOCK_SIZE : IMAGE_HEIGHT - 1;

		auto averageValue = Util::CalculateAverageValue(currentGrayFrame, boundingBoxLeftTopX, boundingBoxLeftTopY, boundingBoxRightBottomX, boundingBoxRightBottomY);

		auto maxdiffBlockRightBottomX = (elem.blockX + 1) * BLOCK_SIZE > IMAGE_WIDTH ? IMAGE_WIDTH - 1 : (elem.blockX + 1) * BLOCK_SIZE;
		auto maxdiffBlockRightBottomY = (elem.blockY + 1) * BLOCK_SIZE > IMAGE_HEIGHT ? IMAGE_HEIGHT - 1 : (elem.blockY + 1) * BLOCK_SIZE;

		for (auto r = elem.blockY * BLOCK_SIZE; r < maxdiffBlockRightBottomY; ++r)
		{
			auto ptr = currentGrayFrame.ptr<uchar>(r);
			for (auto c = elem.blockX * BLOCK_SIZE; c < maxdiffBlockRightBottomX; ++c)
			{
				if (ptr[c] > averageValue)
				{
					ptr[c] = ptr[c] + 10 > 255 ? 255 : ptr[c] + 10;
				}
			}
		}
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::GetMaxValueOfMatrix(std::vector<std::vector<uchar>> maxmindiff, DifferenceElem& diffElem)
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

inline std::vector<DifferenceElem> DetectByMaxFilterAndAdptiveThreshold::GetMostMaxDiffBlock(std::vector<std::vector<uchar>> maxmindiff)
{
	std::vector<DifferenceElem> mostPossibleBlocks;

//	DifferenceElem diffElem;
//	GetMaxValueOfMatrix(maxmindiff, diffElem);
//	mostPossibleBlocks.push_back(diffElem);

	GetDiffValueOfMatrixBigThanThreshold(maxmindiff, mostPossibleBlocks);

	return mostPossibleBlocks;
}

inline void DetectByMaxFilterAndAdptiveThreshold::SearchNeighbors(const std::vector<std::vector<unsigned char>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal)
{
	auto threshold = 2;

	std::stack<cv::Point> deepTrace;
	deepTrace.push(cv::Point(bc, br));

	while (deepTrace.empty() != true)
	{
		auto top = deepTrace.top();
		deepTrace.pop();

		auto c = top.x;
		auto r = top.y;

		if (r - 1 >= 0 && flag[r - 1][c] == false && abs(static_cast<int>(maxmindiff[r - 1][c]) - diffVal) < threshold)
		{
			flag[r - 1][c] = true;
			deepTrace.push(cv::Point(c, r - 1));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r - 1][c];
			elem.blockX = c;
			elem.blockY = r - 1;
			diffElemVec.push_back(elem);
		}
		if (r + 1 < countY && flag[r + 1][c] == false && abs(static_cast<int>(maxmindiff[r + 1][c]) - diffVal) < threshold)
		{
			flag[r + 1][c] = true;
			deepTrace.push(cv::Point(c, r + 1));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r + 1][c];
			elem.blockX = c;
			elem.blockY = r + 1;
			diffElemVec.push_back(elem);
		}
		if (c - 1 >= 0 && flag[r][c - 1] == false && abs(static_cast<int>(maxmindiff[r][c - 1]) - diffVal) < threshold)
		{
			flag[r][c - 1] = true;
			deepTrace.push(cv::Point(c - 1, r));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r][c - 1];
			elem.blockX = c - 1;
			elem.blockY = r;
			diffElemVec.push_back(elem);
		}
		if (c + 1 < countX && flag[r][c + 1] == false && abs(static_cast<int>(maxmindiff[r][c + 1]) - diffVal) < threshold)
		{
			flag[r][c + 1] = true;
			deepTrace.push(cv::Point(c + 1, r));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r][c + 1];
			elem.blockX = c + 1;
			elem.blockY = r;
			diffElemVec.push_back(elem);
		}
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<uchar>> maxmindiff, std::vector<DifferenceElem>& diffElemVec)
{

	std::vector< std::vector<bool>> flag(countY, std::vector<bool>(countX, false));
	diffElemVec.clear();
	for (auto br = 0; br < countY; ++br)
	{
		for (auto bc = 0; bc < countX; ++bc)
		{
			if (LowContrastThreshold <= static_cast<int>(maxmindiff[br][bc]))
			{
				DifferenceElem diffElem;
				diffElem.blockX = bc;
				diffElem.blockY = br;
				diffElem.diffVal = static_cast<int>(maxmindiff[br][bc]);
				diffElemVec.push_back(diffElem);

				flag[br][bc] = true;

				SearchNeighbors(maxmindiff, diffElemVec, flag, br, bc, static_cast<int>(maxmindiff[br][bc]));
			}
		}
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize)
{
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	dilate(curFrame, filtedFrame, kernel);
}

inline int DetectByMaxFilterAndAdptiveThreshold::GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap)
{
	auto currentIndex = 0;
	for (auto r = 0; r < filtedFrame.rows; ++r)
	{
		auto frameRowPtr = filtedFrame.ptr<uchar>(r);
		auto maskRowPtr = blockMap.ptr<int>(r);
		for (auto c = 0; c < filtedFrame.cols; ++c)
		{
			if (maskRowPtr[c] != -1)
				continue;

			auto val = frameRowPtr[c];
			Util::FindNeighbor(filtedFrame, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	return currentIndex;
}

inline void DetectByMaxFilterAndAdptiveThreshold::Discretization(const cv::Mat& filtedFrame, cv::Mat& discretizatedFrame)
{
	for (auto r = 0; r < filtedFrame.rows; ++r)
	{
		auto srcImgPtr = filtedFrame.ptr<uchar>(r);
		auto destImgPtr = discretizatedFrame.ptr<uchar>(r);

		for (auto c = 0; c < filtedFrame.cols; ++c)
		{
			destImgPtr[c] = (srcImgPtr[c] / DISCRATED_BIN) * DISCRATED_BIN;
		}
	}
}

template<typename DataType>
std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold::Detect(cv::Mat& currentGrayFrame, cv::Mat& preprocessResultFrame, cv::Mat& detectedResultFrame)
{

	StrengthenIntensityOfBlock(currentGrayFrame);

	cv::Mat frameAfterMaxFilter(cv::Size(currentGrayFrame.cols, currentGrayFrame.rows), CV_8UC1);
	MaxFilter(currentGrayFrame, frameAfterMaxFilter, DilateKernelSize);

	cv::Mat frameAfterDiscrezated(cv::Size(currentGrayFrame.cols, currentGrayFrame.rows), CV_8UC1);
	Discretization(frameAfterMaxFilter, frameAfterDiscrezated);

	preprocessResultFrame = frameAfterDiscrezated;

	imshow("Max Filter and Discrezated", frameAfterDiscrezated);

	cv::Mat blockMap(cv::Size(frameAfterDiscrezated.cols, frameAfterDiscrezated.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks(frameAfterDiscrezated, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects);

//	Util::ShowAllObject(currentGrayFrame, allObjects, "All Rectangles Checked by Mask");

	RemoveSmallAndBigObjects(allObjects);

//	Util::ShowAllObject(currentGrayFrame, allObjects, "After Remove Rect out range size");

	RemoveObjectsWithLowContrast(allObjects, frameAfterDiscrezated);

//	Util::ShowAllObject(frameAfterDiscrezated, allObjects, "After Remove Low contrast");

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(allObjects, afterMergeObjects);

	Util::ShowAllObject(frameAfterDiscrezated, afterMergeObjects, "After Merge Cross Rectangles");

	DoubleCheckAfterMerge(frameAfterDiscrezated, afterMergeObjects);

	auto rects = Util::GetCandidateTargets(afterMergeObjects);

	Util::ShowAllCandidateTargets(currentGrayFrame, rects);

	return rects;
}