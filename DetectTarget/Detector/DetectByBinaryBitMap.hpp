#pragma once
#include <core/core.hpp>
#include "../Utils/Util.hpp"
#include "../Models/FourLimits.hpp"
#include "../Models/FieldType.hpp"

class DetectByBinaryBitMap
{
public:
	
	static void Detect(cv::Mat curFrame);

private:

	static int GetBitMap(const cv::Mat& binaryFrame, cv::Mat& bitMap);

};

inline void DetectByBinaryBitMap::Detect(cv::Mat curFrame)
{
	cv::Mat binaryFrame;
	curFrame.copyTo(binaryFrame);
	Util::BinaryMat(binaryFrame);

	cv::Mat bitMap(cv::Size(binaryFrame.cols, binaryFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBitMap(binaryFrame, bitMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(bitMap, allObjects);

	Util::ShowAllObject(curFrame, allObjects);
	Util::ShowAllCandidateTargets(curFrame, allObjects);
}

inline int DetectByBinaryBitMap::GetBitMap(const cv::Mat& binaryFrame, cv::Mat& bitMap)
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

			Util::FindNeighbor(binaryFrame, bitMap, r, c, currentIndex++, FieldType::Eight);
		}
	}
	return currentIndex;
}

