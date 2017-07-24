#pragma once
#include <core/core.hpp>
#include "../Utils/Util.hpp"
#include "../Models/FourLimits.hpp"
#include "../Models/FieldType.hpp"

template <typename DataType>
class DetectByBinaryBitMap
{
public:
	static void Detect(cv::Mat frame);

private:
	static int GetBitMap(const cv::Mat& binaryFrame, cv::Mat& bitMap);
};

template <typename DataType>
void DetectByBinaryBitMap<DataType>::Detect(cv::Mat frame)
{
	cv::Mat binaryFrame;
	frame.copyTo(binaryFrame);
	Util<DataType>::BinaryMat(binaryFrame);

	cv::Mat bitMap(cv::Size(binaryFrame.cols, binaryFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBitMap(binaryFrame, bitMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util<DataType>::GetRectangleSize(bitMap, allObjects);

	Util<DataType>::ShowAllObject(frame, allObjects);
	Util<DataType>::ShowAllCandidateTargets(frame, allObjects);
}

template <typename DataType>
int DetectByBinaryBitMap<DataType>::GetBitMap(const cv::Mat& binaryFrame, cv::Mat& bitMap)
{
	auto currentIndex = 0;
	for (auto r = 0; r < binaryFrame.rows; ++r)
	{
		for (auto c = 0; c < binaryFrame.cols; ++c)
		{
			if (binaryFrame.at<uchar>(r, c) == 1)
				continue;
			if (bitMap.at<int32_t>(r, c) != -1)
				continue;

			Util<DataType>::FindNeighbor(binaryFrame, bitMap, r, c, currentIndex++, FieldType::Eight);
		}
	}
	return currentIndex;
}
