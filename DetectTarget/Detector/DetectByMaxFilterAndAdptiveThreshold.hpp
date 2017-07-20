#pragma once
#include <core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include "../Models/FourLimits.hpp"
#include "../Utils/Util.hpp"
#include "../DifferenceElem.hpp"
#include "../Utils/PerformanceUtil.hpp"

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

	template <typename DataType>
	std::vector<cv::Rect> Detect(cv::Mat& curFrame, cv::Mat& preprocessResultFrame);

private:

	void RefreshBlockMap();

	void MaxFilter(int kernelSize);

	template <typename DataType>
	void GetBlocks();

	template <typename DataType>
	void Discretization();

	void MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects) const;

	template <typename DataType>
	std::vector<std::vector<DataType>> GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame);

	template <typename DataType>
	void StrengthenIntensityOfBlock();

	template <typename DataType>
	std::vector<DifferenceElem> GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff);

	template <typename DataType>
	void SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal);

	template <typename DataType>
	void GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec);

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const;

	void RemoveSmallAndBigObjects();

	template <typename DataType>
	void RemoveObjectsWithLowContrast(std::vector<FourLimits>& allObjects, const cv::Mat& frame) const;

	template <typename DataType>
	void DoubleCheckAfterMerge(const cv::Mat& frame, std::vector<FourLimits>& allObjects);

private:

	int imageWidth;
	int imageHeight;

	int totalObject;
	std::vector<FourLimits> fourLimitsOfAllObjects;

	cv::Mat frameNeedDetect;
	cv::Mat frameAfterMaxFilter;
	cv::Mat frameAfterDiscrezated;
	cv::Mat blockMap;
};

template <typename DataType>
std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold::Detect(cv::Mat& currentGrayFrame, cv::Mat& preprocessResultFrame)
{
	frameNeedDetect = currentGrayFrame;
	RefreshBlockMap();
	totalObject = 0;
	fourLimitsOfAllObjects.clear();

	StrengthenIntensityOfBlock<DataType>();

	MaxFilter(DilateKernelSize);

	Discretization<DataType>();

	GetBlocks<DataType>();
	fourLimitsOfAllObjects.resize(totalObject);

	Util::GetRectangleSize(blockMap, fourLimitsOfAllObjects);

	RemoveSmallAndBigObjects();

	RemoveObjectsWithLowContrast<DataType>(fourLimitsOfAllObjects, frameAfterDiscrezated);

	std::vector<FourLimits> afterMergeObjects;
	MergeCrossedRectangles(fourLimitsOfAllObjects, afterMergeObjects);

	DoubleCheckAfterMerge<DataType>(frameAfterDiscrezated, afterMergeObjects);

	auto rects = Util::GetCandidateTargets(afterMergeObjects);

	preprocessResultFrame = frameAfterDiscrezated;

	return rects;
}

inline bool DetectByMaxFilterAndAdptiveThreshold::CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const
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

inline void DetectByMaxFilterAndAdptiveThreshold::RemoveSmallAndBigObjects()
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
void DetectByMaxFilterAndAdptiveThreshold::RemoveObjectsWithLowContrast(std::vector<FourLimits>& allObjects, const cv::Mat& frame) const
{
	for (auto it = allObjects.begin(); it != allObjects.end();)
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
		if (rightBottomX > frame.cols)
		{
			rightBottomX = frame.cols;
		}

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY > frame.rows)
		{
			rightBottomY = frame.rows;
		}

		Util::CalculateThreshHold<DataType>(frame, threshold, leftTopX, leftTopY, rightBottomX, rightBottomY);

		Util::CalCulateCenterValue<DataType>(frame, centerValue, cv::Rect(it->left, it->top, it->right - it->left + 1, it->bottom - it->top + 1));

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

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::DoubleCheckAfterMerge(const cv::Mat& frame, std::vector<FourLimits>& allObjects)
{
	RemoveSmallAndBigObjects();
	RemoveObjectsWithLowContrast<DataType>(allObjects, frame);
}

inline void DetectByMaxFilterAndAdptiveThreshold::MergeCrossedRectangles(std::vector<FourLimits>& allObjects, std::vector<FourLimits>& afterMergeObjects) const
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

template <typename DataType>
std::vector<std::vector<DataType>> DetectByMaxFilterAndAdptiveThreshold::GetMaxMinPixelValueDifferenceMap(cv::Mat& curFrame)
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

template <typename DataType>
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

		auto averageValue = Util::CalculateAverageValue<DataType>(frameNeedDetect, boundingBoxLeftTopX, boundingBoxLeftTopY, boundingBoxRightBottomX, boundingBoxRightBottomY);

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
std::vector<DifferenceElem> DetectByMaxFilterAndAdptiveThreshold::GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff)
{
	std::vector<DifferenceElem> mostPossibleBlocks;

	GetDiffValueOfMatrixBigThanThreshold<DataType>(maxmindiff, mostPossibleBlocks);

	return mostPossibleBlocks;
}

template <typename DataType>
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

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec)
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

				SearchNeighbors<DataType>(maxmindiff, diffElemVec, flag, br, bc, static_cast<int>(maxmindiff[br][bc]));
			}
		}
	}
}

inline void DetectByMaxFilterAndAdptiveThreshold::RefreshBlockMap()
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

inline void DetectByMaxFilterAndAdptiveThreshold::MaxFilter(int kernelSize)
{
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	dilate(frameNeedDetect, frameAfterMaxFilter, kernel);
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::GetBlocks()
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
			Util::FindNeighbor<DataType>(frameAfterDiscrezated, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	totalObject = currentIndex;
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold::Discretization()
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
