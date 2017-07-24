#pragma once

#include <opencv2/core/core.hpp>
#include "../Headers/GlobalConstantConfigure.h"
#include "../Utils/Util.hpp"

const auto CONTIUNITY_THRESHHOLD = 0.4;

template <typename DataType>
class DetectByDiscontinuity
{
public:
	static void Detect(cv::Mat frame);

private:
	static bool CheckDiscontinuity(const cv::Mat& frame, const cv::Point& leftTop);
};

template <typename DataType>
void DetectByDiscontinuity<DataType>::Detect(cv::Mat frame)
{
	std::vector<cv::Rect> candidateRects;

	for (auto r = 0; r < frame.rows - SEARCH_WINDOW_HEIGHT + 1; ++r)
	{
		for (auto c = 0; c < frame.cols - SEARCH_WINDOW_WIDTH + 1; ++c)
		{
			if (CheckDiscontinuity(frame, cv::Point(c, r)))
				candidateRects.push_back(cv::Rect(c, r, SEARCH_WINDOW_WIDTH, SEARCH_WINDOW_HEIGHT));
		}
	}

	Util<DataType>::ShowCandidateRects(frame, candidateRects);
}

template <typename DataType>
bool DetectByDiscontinuity<DataType>::CheckDiscontinuity(const cv::Mat& frame, const cv::Point& leftTop)
{
	auto curRect = cv::Rect(leftTop.x, leftTop.y, SEARCH_WINDOW_WIDTH, SEARCH_WINDOW_HEIGHT);
	cv::Mat curMat;
	frame(curRect).copyTo(curMat);

	auto regionMean = Util<DataType>::MeanMat(curMat);

	Util<DataType>::BinaryMat(curMat);

	auto rowTop = leftTop.y - 1;
	auto rowBottom = leftTop.y + SEARCH_WINDOW_HEIGHT;
	auto colLeft = leftTop.x - 1;
	auto colRight = leftTop.x + SEARCH_WINDOW_WIDTH;

	auto totalCount = 0;
	auto continuityCount = 0;

	auto sum = 0.0;

	if (rowTop >= 0)
	{
		for (auto x = leftTop.x; x < leftTop.x + SEARCH_WINDOW_WIDTH; ++x)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(rowTop, x));

			auto curValue = frame.at<uchar>(rowTop, x) > THRESHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(rowTop - leftTop.y + 1, x - leftTop.x))
				continuityCount++;
		}
	}
	if (rowBottom < frame.rows)
	{
		for (auto x = leftTop.x; x < leftTop.x + SEARCH_WINDOW_WIDTH; ++x)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(rowBottom, x));

			auto curValue = frame.at<uchar>(rowBottom, x) > THRESHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(rowBottom - leftTop.y - 1, x - leftTop.x))
				continuityCount++;
		}
	}

	if (colLeft >= 0)
	{
		for (auto y = leftTop.y; y < leftTop.y + SEARCH_WINDOW_HEIGHT; ++y)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(y, colLeft));

			auto curValue = frame.at<uchar>(y, colLeft) > THRESHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(y - leftTop.y, colLeft - leftTop.x + 1))
				continuityCount++;
		}
	}

	if (colRight < frame.cols)
	{
		for (auto y = leftTop.y; y < leftTop.y + SEARCH_WINDOW_HEIGHT; ++y)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(y, colRight));

			auto curValue = frame.at<uchar>(y, colRight) > THRESHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(y - leftTop.y, colRight - leftTop.x - 1))
				continuityCount++;
		}
	}

	auto roundMean = sum / totalCount;

	return std::abs(roundMean - regionMean) > 2 && regionMean > THRESHOLD;

	//	return static_cast<double>(continuityCount) / totalCount < CONTIUNITY_THRESHHOLD;
}
