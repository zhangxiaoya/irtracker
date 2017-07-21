#pragma once
#include <core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include "../Models/FourLimits.hpp"
#include "../Utils/Util.hpp"
#include "../Utils/PerformanceUtil.hpp"
#include "../Models/DifferenceElem.hpp"

template <class DataType>
class DetectByMaxFilterAndAdptiveThreshold
{
public:

	DetectByMaxFilterAndAdptiveThreshold(int image_width, int image_height)
		: imageWidth(image_width),
		  imageHeight(image_height),
		  totalObject(0)
	{
		frameAfterMaxFilter = Mat(imageHeight, imageWidth, CV_DATA_TYPE);
		frameAfterDiscrezated = Mat(imageHeight, imageWidth, CV_DATA_TYPE);
		blockMap = Mat(imageHeight, imageWidth, CV_32SC1, cv::Scalar(-1));
	}

	void Reset(cv::Mat& currentGrayFrame);

	std::vector<cv::Rect> Detect(cv::Mat& curFrame, cv::Mat& preprocessResultFrame);

private:

	void RefreshBlockMap();

	void MaxFilter(int kernelSize);
	
	void GetBlocks();

	void Discretization();

	void MergeCrossedRectangles();

	std::vector<std::vector<DataType>> GetMaxMinPixelValueDifferenceMap();

	void StrengthenIntensityOfBlock();

	std::vector<DifferenceElem> GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff);

	void SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal);

	void GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec);

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const;

	void RemoveSmallAndBigObjects();

	void RemoveObjectsWithLowContrast();

	void DoubleCheckAfterMerge();

private:

	int imageWidth;
	int imageHeight;

	int totalObject;
	std::vector<FourLimits> fourLimitsOfAllObjects;
	std::vector<FourLimits> fourLimitsAfterMergeObjects;

	cv::Mat frameNeedDetect;
	cv::Mat frameAfterMaxFilter;
	cv::Mat frameAfterDiscrezated;
	cv::Mat blockMap;
};

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::Reset(cv::Mat& currentGrayFrame)
{
	frameNeedDetect = currentGrayFrame;

	RefreshBlockMap();
	totalObject = 0;
	fourLimitsOfAllObjects.clear();
	fourLimitsAfterMergeObjects.clear();
}

template <typename DataType>
std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold<DataType>::Detect(cv::Mat& currentGrayFrame, cv::Mat& preprocessResultFrame)
{
	Reset(currentGrayFrame);

	StrengthenIntensityOfBlock();

	MaxFilter(DilateKernelSize);

	Discretization();

	GetBlocks();

	Util<DataType>::GetRectangleSize(blockMap, fourLimitsOfAllObjects);

	RemoveSmallAndBigObjects();

	RemoveObjectsWithLowContrast();

	MergeCrossedRectangles();

	DoubleCheckAfterMerge();

	auto rects = Util<DataType>::GetCandidateTargets(fourLimitsAfterMergeObjects);

	preprocessResultFrame = frameAfterDiscrezated;

	return rects;
}

template <typename DataType>
bool DetectByMaxFilterAndAdptiveThreshold<DataType>::CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const
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

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::RemoveSmallAndBigObjects()
{
	for (auto it = fourLimitsOfAllObjects.begin(); it != fourLimitsOfAllObjects.end();)
	{
		auto width = it->right - it->left + 1;
		auto height = it->bottom - it->top + 1;

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			it = fourLimitsOfAllObjects.erase(it);
		else
			++it;
	}
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::RemoveObjectsWithLowContrast()
{
	for (auto it = fourLimitsOfAllObjects.begin(); it != fourLimitsOfAllObjects.end();)
	{
		DataType threshold = 0;
		DataType centerValue = 0;

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
		if (rightBottomX > imageWidth)
		{
			rightBottomX = imageWidth;
		}

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY > imageHeight)
		{
			rightBottomY = imageHeight;
		}

		Util<DataType>::CalculateThreshHold(frameAfterDiscrezated, threshold, leftTopX, leftTopY, rightBottomX, rightBottomY);

		Util<DataType>::CalCulateCenterValue(frameAfterDiscrezated, centerValue, cv::Rect(it->left, it->top, it->right - it->left + 1, it->bottom - it->top + 1));

		if (std::abs(static_cast<int>(centerValue) - static_cast<int>(threshold)) < 3)
		{
			it = fourLimitsOfAllObjects.erase(it);
		}
		else
		{
			++it;
		}
	}
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::DoubleCheckAfterMerge()
{
	RemoveSmallAndBigObjects();
	RemoveObjectsWithLowContrast();
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::MergeCrossedRectangles()
{
	for (auto i = 0; i < fourLimitsOfAllObjects.size(); ++i)
	{
		if (fourLimitsOfAllObjects[i].identify == -1)
			continue;
		for (auto j = 0; j < fourLimitsOfAllObjects.size(); ++j)
		{
			if (i == j || fourLimitsOfAllObjects[j].identify == -1)
				continue;
			if (CheckCross(fourLimitsOfAllObjects[i], fourLimitsOfAllObjects[j]))
			{
				fourLimitsOfAllObjects[j].identify = -1;

				if (fourLimitsOfAllObjects[i].top > fourLimitsOfAllObjects[j].top)
					fourLimitsOfAllObjects[i].top = fourLimitsOfAllObjects[j].top;

				if (fourLimitsOfAllObjects[i].left > fourLimitsOfAllObjects[j].left)
					fourLimitsOfAllObjects[i].left = fourLimitsOfAllObjects[j].left;

				if (fourLimitsOfAllObjects[i].right < fourLimitsOfAllObjects[j].right)
					fourLimitsOfAllObjects[i].right = fourLimitsOfAllObjects[j].right;

				if (fourLimitsOfAllObjects[i].bottom < fourLimitsOfAllObjects[j].bottom)
					fourLimitsOfAllObjects[i].bottom = fourLimitsOfAllObjects[j].bottom;
			}
		}
	}
	// for left top may be missed, so need double check
	for (auto i = 0; i < fourLimitsOfAllObjects.size(); ++i)
	{
		if (fourLimitsOfAllObjects[i].identify == -1)
			continue;
		for (auto j = 0; j < fourLimitsOfAllObjects.size(); ++j)
		{
			if (i == j || fourLimitsOfAllObjects[j].identify == -1)
				continue;
			if (CheckCross(fourLimitsOfAllObjects[i], fourLimitsOfAllObjects[j]))
			{
				fourLimitsOfAllObjects[j].identify = -1;

				if (fourLimitsOfAllObjects[i].top > fourLimitsOfAllObjects[j].top)
					fourLimitsOfAllObjects[i].top = fourLimitsOfAllObjects[j].top;

				if (fourLimitsOfAllObjects[i].left > fourLimitsOfAllObjects[j].left)
					fourLimitsOfAllObjects[i].left = fourLimitsOfAllObjects[j].left;

				if (fourLimitsOfAllObjects[i].right < fourLimitsOfAllObjects[j].right)
					fourLimitsOfAllObjects[i].right = fourLimitsOfAllObjects[j].right;

				if (fourLimitsOfAllObjects[i].bottom < fourLimitsOfAllObjects[j].bottom)
					fourLimitsOfAllObjects[i].bottom = fourLimitsOfAllObjects[j].bottom;
			}
		}
	}

	for (auto i = 0; i < fourLimitsOfAllObjects.size(); ++i)
	{
		if (fourLimitsOfAllObjects[i].identify != -1)
			fourLimitsAfterMergeObjects.push_back(fourLimitsOfAllObjects[i]);
	}
}

template <typename DataType>
std::vector<std::vector<DataType>> DetectByMaxFilterAndAdptiveThreshold<DataType>::GetMaxMinPixelValueDifferenceMap()
{
	std::vector<std::vector<DataType>> maxmindiff(countY, std::vector<DataType>(countX, static_cast<DataType>(0)));
	for (auto br = 0; br < countY; ++br)
	{
		auto height = br == (countY - 1) ? IMAGE_HEIGHT - (countY - 1) * BLOCK_SIZE : BLOCK_SIZE;
		for (auto bc = 0; bc < countX; ++bc)
		{
			auto width = bc == (countX - 1) ? IMAGE_WIDTH - (countX - 1) * BLOCK_SIZE : BLOCK_SIZE;
			maxmindiff[br][bc] =
				Util<DataType>::GetMaxValueOfBlock(frameNeedDetect(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height))) -
				Util<DataType>::GetMinValueOfBlock(frameNeedDetect(cv::Rect(bc * BLOCK_SIZE, br * BLOCK_SIZE, width, height)));
		}
	}
	return maxmindiff;
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::StrengthenIntensityOfBlock()
{
	auto maxmindiffMatrix = GetMaxMinPixelValueDifferenceMap();

	auto differenceElems = GetMostMaxDiffBlock(maxmindiffMatrix);

	for (auto elem : differenceElems)
	{
		auto centerX = elem.blockX * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto centerY = elem.blockY * BLOCK_SIZE + BLOCK_SIZE / 2;
		auto boundingBoxLeftTopX = centerX - BLOCK_SIZE >= 0 ? centerX - BLOCK_SIZE : 0;
		auto boundingBoxLeftTopY = centerY - BLOCK_SIZE >= 0 ? centerY - BLOCK_SIZE : 0;
		auto boundingBoxRightBottomX = centerX + BLOCK_SIZE < IMAGE_WIDTH ? centerX + BLOCK_SIZE : IMAGE_WIDTH - 1;
		auto boundingBoxRightBottomY = centerY + BLOCK_SIZE < IMAGE_HEIGHT ? centerY + BLOCK_SIZE : IMAGE_HEIGHT - 1;

		auto averageValue = Util<DataType>::CalculateAverageValue(frameNeedDetect, boundingBoxLeftTopX, boundingBoxLeftTopY, boundingBoxRightBottomX, boundingBoxRightBottomY);

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

template <typename DataType>
std::vector<DifferenceElem> DetectByMaxFilterAndAdptiveThreshold<DataType>::GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff)
{
	std::vector<DifferenceElem> mostPossibleBlocks;

	GetDiffValueOfMatrixBigThanThreshold(maxmindiff, mostPossibleBlocks);

	return mostPossibleBlocks;
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal)
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

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec)
{
	std::vector<std::vector<bool>> flag(countY, std::vector<bool>(countX, false));
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

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::RefreshBlockMap()
{
	for (auto r = 0; r < imageHeight; ++r)
	{
		auto ptr = blockMap.ptr<int>(r);
		for (auto c = 0; c < imageWidth; ++c)
		{
			ptr[c] = -1;
		}
	}
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::MaxFilter(int kernelSize)
{
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	dilate(frameNeedDetect, frameAfterMaxFilter, kernel);
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::GetBlocks()
{
	auto currentIndex = 0;
	for (auto r = 0; r < imageHeight; ++r)
	{
		auto frameRowPtr = frameAfterDiscrezated.ptr<DataType>(r);
		auto maskRowPtr = blockMap.ptr<int>(r);
		for (auto c = 0; c < imageWidth; ++c)
		{
			if (maskRowPtr[c] != -1)
				continue;

			auto val = frameRowPtr[c];
			Util<DataType>::FindNeighbor(frameAfterDiscrezated, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	totalObject = currentIndex;
	fourLimitsOfAllObjects.resize(totalObject);
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::Discretization()
{
	for (auto r = 0; r < frameAfterMaxFilter.rows; ++r)
	{
		auto srcImgPtr = frameAfterMaxFilter.ptr<DataType>(r);
		auto destImgPtr = frameAfterDiscrezated.ptr<DataType>(r);

		for (auto c = 0; c < frameAfterMaxFilter.cols; ++c)
		{
			destImgPtr[c] = (srcImgPtr[c] / DISCRATED_BIN) * DISCRATED_BIN;
		}
	}
}
