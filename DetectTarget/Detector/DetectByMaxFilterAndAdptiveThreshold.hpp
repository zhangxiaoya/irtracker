#pragma once
#include <core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include "../Models/FourLimits.hpp"
#include "../Utils/Util.hpp"
#include "../Utils/PerformanceUtil.hpp"
#include "../PreProcessor/PreProcesssorFactory.hpp"

template <class DataType>
class DetectByMaxFilterAndAdptiveThreshold
{
public:

	DetectByMaxFilterAndAdptiveThreshold(int image_width, int image_height)
		: imageWidth(image_width),
		  imageHeight(image_height),
		  totalObject(0)
	{
		preprocessor = PreProcessorFactory::CreatePreProcessor<DataType>(image_width, image_height);
		frameAfterMaxFilter = Mat(imageHeight, imageWidth, CV_DATA_TYPE);
		frameAfterDiscrezated = Mat(imageHeight, imageWidth, CV_DATA_TYPE);
		blockMap = Mat(imageHeight, imageWidth, CV_32SC1, cv::Scalar(-1));
	}

	void Reload(cv::Mat& currentGrayFrame);

	std::vector<cv::Rect> Detect(cv::Mat& curFrame, cv::Mat& preprocessResultFrame);

private:

	void RefreshBlockMap();

	void MaxFilter(int kernelSize);

	void GetBlocks();

	void Discretization();

	void MergeCrossedRectangles();

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const;

	void RemoveSmallAndBigObjects(std::vector<FourLimits>& objects);

	void RemoveObjectsWithLowContrast(std::vector<FourLimits>& objects);

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
	cv::Ptr<PreProcessor<DataType>> preprocessor;
};

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::Reload(cv::Mat& currentGrayFrame)
{
	frameNeedDetect = currentGrayFrame;

	RefreshBlockMap();
	totalObject = 0;
	fourLimitsOfAllObjects.clear();
	fourLimitsAfterMergeObjects.clear();

	preprocessor->SetSourceFrame(frameNeedDetect);
}

template <typename DataType>
std::vector<cv::Rect> DetectByMaxFilterAndAdptiveThreshold<DataType>::Detect(cv::Mat& currentGrayFrame, cv::Mat& preprocessResultFrame)
{
	Reload(currentGrayFrame);

	preprocessor->StrengthenIntensityOfBlock();

	MaxFilter(DilateKernelSize);

	Discretization();

	GetBlocks();

	Util<DataType>::GetRectangleSize(blockMap, fourLimitsOfAllObjects);

	RemoveSmallAndBigObjects(fourLimitsOfAllObjects);

	RemoveObjectsWithLowContrast(fourLimitsOfAllObjects);

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
void DetectByMaxFilterAndAdptiveThreshold<DataType>::RemoveSmallAndBigObjects(std::vector<FourLimits>& objects)
{
	for (auto it = objects.begin(); it != objects.end();)
	{
		auto width = it->right - it->left + 1;
		auto height = it->bottom - it->top + 1;

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			it = objects.erase(it);
		else
		{
			++it;
			auto dummy = 1;
		}
	}
}

template <typename DataType>
void DetectByMaxFilterAndAdptiveThreshold<DataType>::RemoveObjectsWithLowContrast(std::vector<FourLimits>& objects)
{
	for (auto it = objects.begin(); it != objects.end();)
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
			it = objects.erase(it);
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
	RemoveSmallAndBigObjects(fourLimitsAfterMergeObjects);
	RemoveObjectsWithLowContrast(fourLimitsAfterMergeObjects);
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
