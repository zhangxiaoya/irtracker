#pragma once

#include <opencv2/core/core.hpp>
#include "Util.hpp"

const auto CONTIUNITY_THRESHHOLD = 0.4;

class DetectByDiscontinuity
{
public:

	static void Detect(cv::Mat curFrame);

private:

	static bool CheckDiscontinuity(const cv::Mat& cur_frame, const cv::Point& leftTop);

};

inline bool DetectByDiscontinuity::CheckDiscontinuity(const cv::Mat& frame, const cv::Point& leftTop)
{
	auto curRect = cv::Rect(leftTop.x, leftTop.y, WINDOW_WIDTH, WINDOW_HEIGHT);
	cv::Mat curMat;
	frame(curRect).copyTo(curMat);

	auto regionMean = Util::MeanMat(curMat);

	Util::BinaryMat(curMat);

	auto rowTop = leftTop.y - 1;
	auto rowBottom = leftTop.y + WINDOW_HEIGHT;
	auto colLeft = leftTop.x - 1;
	auto colRight = leftTop.x + WINDOW_WIDTH;

	auto totalCount = 0;
	auto continuityCount = 0;

	auto sum = 0.0;

	if (rowTop >= 0)
	{
		for (auto x = leftTop.x; x < leftTop.x + WINDOW_WIDTH; ++x)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(rowTop, x));

			auto curValue = frame.at<uchar>(rowTop, x) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(rowTop - leftTop.y + 1, x - leftTop.x))
				continuityCount++;
		}
	}
	if (rowBottom < frame.rows)
	{
		for (auto x = leftTop.x; x < leftTop.x + WINDOW_WIDTH; ++x)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(rowBottom, x));

			auto curValue = frame.at<uchar>(rowBottom, x) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(rowBottom - leftTop.y - 1, x - leftTop.x))
				continuityCount++;
		}
	}

	if (colLeft >= 0)
	{
		for (auto y = leftTop.y; y < leftTop.y + WINDOW_HEIGHT; ++y)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(y, colLeft));

			auto curValue = frame.at<uchar>(y, colLeft) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(y - leftTop.y, colLeft - leftTop.x + 1))
				continuityCount++;
		}
	}

	if (colRight < frame.cols)
	{
		for (auto y = leftTop.y; y < leftTop.y + WINDOW_HEIGHT; ++y)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(y, colRight));

			auto curValue = frame.at<uchar>(y, colRight) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(y - leftTop.y, colRight - leftTop.x - 1))
				continuityCount++;
		}
	}

	auto roundMean = sum / totalCount;

	return std::abs(roundMean - regionMean) > 2 && regionMean > THRESHHOLD;

	//	return static_cast<double>(continuityCount) / totalCount < CONTIUNITY_THRESHHOLD;
}

inline void DetectByDiscontinuity::Detect(cv::Mat curFrame)
{
	std::vector<cv::Rect> candidateRects;

	for (auto r = 0; r < curFrame.rows - WINDOW_HEIGHT + 1; ++r)
	{
		for (auto c = 0; c < curFrame.cols - WINDOW_WIDTH + 1; ++c)
		{
			if (CheckDiscontinuity(curFrame, cv::Point(c, r)))
				candidateRects.push_back(cv::Rect(c, r, WINDOW_WIDTH, WINDOW_HEIGHT));
		}
	}

	Util::ShowCandidateRects(curFrame, candidateRects);
}
