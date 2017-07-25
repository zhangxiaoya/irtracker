#pragma once

#include <highgui/highgui.hpp>
#include <iostream>
#include "../Utils/Util.hpp"

template <typename DatType>
class DetectByMultiScaleLocalDifference
{
public:
	static void Detect(cv::Mat curFrame);

private:
	static std::vector<unsigned char>::value_type&& GetAverageGrayValueOfKNeighbor(const cv::Mat& curFrame, int r, int c, int i);
};

template <typename DatType>
void DetectByMultiScaleLocalDifference<DatType>::Detect(cv::Mat curFrame)
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

			auto maxVal = Util<DatType>::MaxOfVector(averageOfKNeighbor, 0, averageOfKNeighbor.size());
			auto minVal = Util<DatType>::MinOfVector(averageOfKNeighbor, 0, averageOfKNeighbor.size());

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

			mldFilterFrame.at<uchar>(r, c) = Util<DatType>::MaxOfVector(contrastOfKNeighbor, 0, contrastOfKNeighbor.size());
		}
	}

	imshow("Map", mldFilterFrame);
}

template <typename DatType>
std::vector<unsigned char>::value_type&& DetectByMultiScaleLocalDifference<DatType>::GetAverageGrayValueOfKNeighbor(const cv::Mat& curFrame, int r, int c, int i)
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
