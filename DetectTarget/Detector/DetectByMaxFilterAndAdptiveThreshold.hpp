#pragma once
#include <core/core.hpp>
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


	DetectByMaxFilterAndAdptiveThreshold(int image_width, int image_height)
		: imageWidth(image_width),
		  imageHeight(image_height)
	{
		frameAfterMaxFilter = Mat(imageHeight, imageWidth, CV_DATA_TYPE);
		frameAfterDiscrezated = Mat(imageHeight, imageWidth, CV_DATA_TYPE);
	}

	template<typename DataType>
	std::vector<cv::Rect> Detect(cv::Mat& curFrame, cv::Mat& preprocessResultFrame);

private:

	void MaxFilter(int kernelSize);

	int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap);

	void Discretization();

	void MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects);

	void RefreshMask(cv::Mat curFrame, std::vector<cv::Rect> result);

	void FilterRectByContinuty(cv::Mat curFrame, std::vector<cv::Rect> rects, std::vector<cv::Rect> result);

	template<typename DataType>
	std::vector<std::vector<uchar>> GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame);

	template<typename DataType>
	void StrengthenIntensityOfBlock();

	void GetMaxValueOfMatrix(std::vector<std::vector<uchar>> maxmindiff, DifferenceElem& diffElem);

	template<typename DataType>
	std::vector<DifferenceElem> GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff);

	template<typename DataType>
	void SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal);

	template<typename DataType>
	void GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec);

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond);

	void CalculateThreshold(const cv::Mat& frame, uchar& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

	void RemoveSmallAndBigObjects(std::vector<FourLimits>& allObjects);

	void RemoveObjectsWithLowContrast(std::vector<FourLimits>& allObjects, const cv::Mat& frame);

	void DoubleCheckAfterMerge(const cv::Mat& frame, std::vector<FourLimits>& allObjects);

	void FillRectToFrame(cv::Rect& rect);

	bool CheckRect(cv::Rect& rect);

private:
	int imageWidth;
	int imageHeight;

	cv::Mat frameNeedDetect;
	cv::Mat frameAfterMaxFilter;
	cv::Mat frameAfterDiscrezated;
};

template<typename DataType>
std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold::Detect(cv::Mat& currentGrayFrame, cv::Mat& preprocessResultFrame)
{
	frameNeedDetect = currentGrayFrame;

	StrengthenIntensityOfBlock<DataType>();

	MaxFilter(DilateKernelSize);

	Discretization();

	preprocessResultFrame = frameAfterDiscrezated;

	cv::Mat blockMap(cv::Size(frameAfterDiscrezated.cols, frameAfterDiscrezated.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks(frameAfterDiscrezated, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects);

	RemoveSmallAndBigObjects(allObjects);

	RemoveObjectsWithLowContrast(allObjects, frameAfterDiscrezated);

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(allObjects, afterMergeObjects);

	DoubleCheckAfterMerge(frameAfterDiscrezated, afterMergeObjects);

	auto rects = Util::GetCandidateTargets(afterMergeObjects);

	return rects;
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

template<typename DataType>
std::vector<std::vector<uchar>> DetectByMaxFilterAndAdptiveThreshold::GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame)
{
	std::vector<std::vector<DataType>> maxmindiff(countY, std::vector<DataType>(countX, static_cast<DataType>(0)));
	for (auto br = 0; br < countY; ++br)
	{
		auto height = br == (countY - 1) ? IMAGE_HEIGHT - (countY - 1) * BLOCK_SIZE : BLOCK_SIZE;
		for (auto bc = 0; bc < countX; ++bc)
		{
			auto width = bc == (countX - 1) ? IMAGE_WIDTH - (countX - 1) * BLOCK_SIZE : BLOCK_SIZE;
			maxmindiff[br][bc] =
				Util::GetMaxValueOfBlock<DataType>(curFrame(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height))) -
				Util::GetMinValueOfBlock<DataType>(curFrame(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height)));
		}
	}
	return maxmindiff;
}

template<typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::StrengthenIntensityOfBlock()
{
	auto maxmindiffMatrix = GetMaxMinPixelValueDifferenceMap<DataType>(frameNeedDetect);

	auto differenceElems = GetMostMaxDiffBlock<DataType>(maxmindiffMatrix);

	for (auto elem : differenceElems)
	{
		auto centerX = elem.blockX * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto centerY = elem.blockY * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto boundingBoxLeftTopX = centerX - BLOCK_SIZE >= 0 ? centerX - BLOCK_SIZE : 0;
		auto boundingBoxLeftTopY = centerY - BLOCK_SIZE >= 0 ? centerY - BLOCK_SIZE : 0;
		auto boundingBoxRightBottomX = centerX + BLOCK_SIZE < IMAGE_WIDTH ? centerX + BLOCK_SIZE : IMAGE_WIDTH - 1;
		auto boundingBoxRightBottomY = centerY + BLOCK_SIZE < IMAGE_HEIGHT ? centerY + BLOCK_SIZE : IMAGE_HEIGHT - 1;

		auto averageValue = Util::CalculateAverageValue(frameNeedDetect, boundingBoxLeftTopX, boundingBoxLeftTopY, boundingBoxRightBottomX, boundingBoxRightBottomY);

		auto maxdiffBlockRightBottomX = (elem.blockX + 1) * BLOCK_SIZE > IMAGE_WIDTH ? IMAGE_WIDTH - 1 : (elem.blockX + 1) * BLOCK_SIZE;
		auto maxdiffBlockRightBottomY = (elem.blockY + 1) * BLOCK_SIZE > IMAGE_HEIGHT ? IMAGE_HEIGHT - 1 : (elem.blockY + 1) * BLOCK_SIZE;

		for (auto r = elem.blockY * BLOCK_SIZE; r < maxdiffBlockRightBottomY; ++r)
		{
			auto ptr = frameNeedDetect.ptr<DataType>(r);
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

template<typename DataType>
std::vector<DifferenceElem> DetectByMaxFilterAndAdptiveThreshold::GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff)
{
	std::vector<DifferenceElem> mostPossibleBlocks;

//	DifferenceElem diffElem;
//	GetMaxValueOfMatrix(maxmindiff, diffElem);
//	mostPossibleBlocks.push_back(diffElem);

	GetDiffValueOfMatrixBigThanThreshold<DataType>(maxmindiff, mostPossibleBlocks);

	return mostPossibleBlocks;
}

template<typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal)
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

template<typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec)
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

				SearchNeighbors<DataType>(maxmindiff, diffElemVec, flag, br, bc, static_cast<int>(maxmindiff[br][bc]));
			}
		}
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::MaxFilter(int kernelSize)
{
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	dilate(frameNeedDetect, frameAfterMaxFilter, kernel);
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

inline void DetectByMaxFilterAndAdptiveThreshold::Discretization()
{
	for (auto r = 0; r < frameAfterMaxFilter.rows; ++r)
	{
		auto srcImgPtr = frameAfterMaxFilter.ptr<uchar>(r);
		auto destImgPtr = frameAfterDiscrezated.ptr<uchar>(r);

		for (auto c = 0; c < frameAfterMaxFilter.cols; ++c)
		{
			destImgPtr[c] = (srcImgPtr[c] / DISCRATED_BIN) * DISCRATED_BIN;
		}
	}
}

