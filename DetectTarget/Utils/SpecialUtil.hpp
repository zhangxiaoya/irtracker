#pragma once
#include <core/core.hpp>

const auto INVALID_PIXEL_ROWS = 2;
const auto INVALID_PIXEL_COLS = 11;

class SpecialUtil
{
public:

	static void RemoveInvalidPixel(cv::Mat curFrame);

	static bool CheckFrameIsGray(const cv::Mat& curFrame, cv::Mat& grayImage);
};

inline void SpecialUtil::RemoveInvalidPixel(cv::Mat curFrame)
{
	for (auto r = 0; r < INVALID_PIXEL_ROWS; ++r)
		for (auto c = 0; c < INVALID_PIXEL_COLS; ++c)
			curFrame.at<uchar>(r, c) = 0;
}

inline bool SpecialUtil::CheckFrameIsGray(const cv::Mat& curFrame, cv::Mat& grayImage)
{
	if (curFrame.channels() == 3)
	{
		cvtColor(curFrame, grayImage, CV_RGB2GRAY);
		return false;
	}
	else
	{
		grayImage = curFrame;
		return true;
	}
}
