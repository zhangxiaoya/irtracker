#pragma once

#include <highgui/highgui.hpp>
#include <iostream>
#include "../Utils/Util.hpp"

class DetectByMultiScaleLocalDifference
{
public:

	static void Detect(cv::Mat curFrame);

private:

	static std::vector<unsigned char>::value_type&& GetAverageGrayValueOfKNeighbor(const cv::Mat& curFrame, int r, int c, int i);

};

inline void DetectByMultiScaleLocalDifference::Detect(cv::Mat curFrame)
{
	cv::Mat mldFilterFrame(cv::Size(curFrame.cols, curFrame.rows), CV_8UC1, cv::Scalar(0));

	std::vector<uchar> averageOfKNeighbor;
	std::vector<uchar> contrastOfKNeighbor;
	auto L = 6;
	for (auto r = 0; r < curFrame.rows; ++r)
	{
		for (auto c = 0; c < curFrame.cols; ++c)
		{
			averageOfKNeighbor.clear();
			contrastOfKNeighbor.clear();

			for (auto i = 1; i <= L; ++i)
				averageOfKNeighbor.push_back(GetAverageGrayValueOfKNeighbor(curFrame, r, c, i));

			auto maxVal = Util::MaxOfVector(averageOfKNeighbor.begin(), averageOfKNeighbor.end());
			auto minVal = Util::MinOfVector(averageOfKNeighbor.begin(), averageOfKNeighbor.end());

			auto squareDiff = (maxVal - minVal) * (maxVal - minVal);

			if (squareDiff == 0)
			{
				mldFilterFrame.at<uchar>(r, c) = maxVal;
				std::cout << "Dummy" << std::endl;
				continue;
			}

			for (auto i = 0; i < L - 1; ++i)
			{
				contrastOfKNeighbor.push_back((averageOfKNeighbor[i] - averageOfKNeighbor[L - 1]) * (averageOfKNeighbor[i] - averageOfKNeighbor[L - 1]) / squareDiff);
			}

			contrastOfKNeighbor.push_back(0);

			mldFilterFrame.at<uchar>(r, c) = Util::MaxOfVector(contrastOfKNeighbor.begin(), contrastOfKNeighbor.end());
		}
	}

	imshow("Map", mldFilterFrame);
}

inline std::vector<unsigned char>::value_type&& DetectByMultiScaleLocalDifference::GetAverageGrayValueOfKNeighbor(const cv::Mat& curFrame, int r, int c, int i)
{
	auto radius = i;
	auto leftTopX = c - i;
	auto leftTopY = r - i;

	auto rightBottomX = leftTopX + 2 * radius;
	auto rightBottomY = leftTopY + 2 * radius;

	auto sum = 0;
	auto totalCount = 0;

	for (auto row = leftTopY; row <= rightBottomY; ++row)
	{
		if (row >= 0 && row < curFrame.rows)
		{
			for (auto col = leftTopX; col <= rightBottomX; ++col)
			{
				if (col >= 0 && col < curFrame.cols)
				{
					sum += curFrame.at<uchar>(row, col);
					++totalCount;
				}
			}
		}
	}
	return sum / totalCount;
}
