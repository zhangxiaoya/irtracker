#pragma once

#include "FourLimits.hpp"
#include "FieldType.hpp"
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ConfidenceElem.hpp"

const auto WINDOW_WIDTH = 8;
const auto WINDOW_HEIGHT = 8;
const auto THRESHHOLD = 25;

auto const REDCOLOR = cv::Scalar(0, 0, 255);
auto const BLUECOLOR = cv::Scalar(255, 0, 0);
auto const GREENCOLOR = cv::Scalar(0, 255, 0);

const auto TARGET_WIDTH_MIN_LIMIT = 2;
const auto TARGET_HEIGHT_MIN_LIMIT = 2;
const auto TARGET_WIDTH_MAX_LIMIT = 16;
const auto TARGET_HEIGHT_MAX_LIMIT = 16;

const auto AFTER_MAX_FILTER = false;

class Util
{
public:

	static void BinaryMat(cv::Mat& mat);

	static double MeanMat(const cv::Mat& mat);

	static void ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects);

	static void FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType, uchar value = 0);

	static void GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject);

	static void ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject);

	static void ShowCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, uchar valueThreshHold = 0);

	static uchar MaxOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end);

	static uchar MinOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end);

	static bool UcharCompare(uchar left, uchar right);

	static bool ConfidenceCompare(ConfidenceElem left, ConfidenceElem right);

	static std::vector<cv::Rect> GetCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& afterMergeObjects, unsigned char max_value);

	static int Sum(std::vector<int>& valueVec);

private:

	static void DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex);

	static void DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, uchar value = 0);

	static void DeepFirstSearch(const cv::Mat& grayFrame, cv::Mat& bitMap, int r, int c, int currentIndex);
};

inline void Util::BinaryMat(cv::Mat& mat)
{
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			mat.at<uchar>(r, c) = mat.at<uchar>(r, c) > THRESHHOLD ? 1 : 0;
}

inline double Util::MeanMat(const cv::Mat& mat)
{
	double sum = 0;
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			sum += static_cast<int>(mat.at<uchar>(r, c));

	return  sum / (mat.rows * mat.cols);
}

inline void Util::ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects)
{
	cv::Mat colorFrame;
	cvtColor(grayFrame, colorFrame, CV_GRAY2RGB);

	for (auto i = 0; i<candidate_rects.size(); ++i)
		rectangle(colorFrame, candidate_rects[i], REDCOLOR);

	imshow("Color Frame", colorFrame);
}

inline void Util::FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType, uchar value)
{
	if (fieldType == FieldType::Eight)
		DFSWithoutRecursionEightField(binaryFrame, bitMap, r, c, currentIndex);
	else if (fieldType == FieldType::Four)
		DFSWithoutRecursionFourField(binaryFrame, bitMap, r, c, currentIndex, value);
	else
		std::cout << "FieldType Error!" << std::endl;
}

inline void Util::GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject)
{
	// top
	for (auto r = 0; r<bitMap.rows; ++r)
	{
		for (auto c = 0; c < bitMap.cols; ++c)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].top == -1)
			{
				allObject[curIndex].top = r;
				if (allObject[curIndex].identify == -1)
					allObject[curIndex].identify = curIndex;
			}
		}
	}
	// bottom
	for (auto r = bitMap.rows - 1; r >= 0; --r)
	{
		for (auto c = 0; c< bitMap.cols; ++c)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].bottom == -1)
				allObject[curIndex].bottom = r;
		}
	}
	// left
	for (auto c = 0; c<bitMap.cols; ++c)
	{
		for (auto r = 0; r <bitMap.rows; ++r)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].left == -1)
				allObject[curIndex].left = c;
		}
	}
	// right
	for (auto c = bitMap.cols - 1; c >= 0; --c)
	{
		for (auto r = 0; r < bitMap.rows; ++r)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].right == -1)
				allObject[curIndex].right = c;
		}
	}
}

inline void Util::ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for (auto i = 0; i<allObject.size(); ++i)
	{
		auto width = allObject[i].right - allObject[i].left + 1;
		auto height = allObject[i].bottom - allObject[i].top + 1;
		if (width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}
		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);
		rectangle(colorFrame, rect, BLUECOLOR);
	}

	imshow("All Object", colorFrame);
}

inline void Util::ShowCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, uchar valueThreshHold)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for (auto i = 0; i<allObject.size(); ++i)
	{
		auto width = allObject[i].right - allObject[i].left + 1;
		auto height = allObject[i].bottom - allObject[i].top + 1;
		if (width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			continue;

		if (curFrame.at<uchar>(allObject[i].top + 1, allObject[i].left + 1) < valueThreshHold)
			continue;

		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);
		rectangle(colorFrame, rect, GREENCOLOR);
	}

	imshow("Candidate Targets", colorFrame);
}

inline uchar Util::MaxOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end)
{
	auto maxResult = *begin;
	for (auto it = begin; it != end; ++it)
	{
		if (maxResult < *it)
			maxResult = *it;
	}
	return maxResult;
}

inline uchar Util::MinOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end)
{
	auto minResult = *begin;
	for (auto it = begin; it != end; ++it)
	{
		if (minResult > *it)
			minResult = *it;
	}
	return minResult;
}

inline bool Util::UcharCompare(uchar left, uchar right)
{
	return left > right;
}

inline bool Util::ConfidenceCompare(ConfidenceElem left, ConfidenceElem right)
{
	return left.confidenceVal > right.confidenceVal;
}

inline std::vector<cv::Rect> Util::GetCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& afterMergeObjects, unsigned char max_value)
{
	std::vector<cv::Rect> targetRect;

	for (auto i = 0; i<afterMergeObjects.size(); ++i)
	{
		auto width = afterMergeObjects[i].right - afterMergeObjects[i].left + 1;
		auto height = afterMergeObjects[i].bottom - afterMergeObjects[i].top + 1;
		if (width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			continue;

		if (curFrame.at<uchar>(afterMergeObjects[i].top + 1, afterMergeObjects[i].left + 1) < max_value)
			continue;

		auto rect = cv::Rect(afterMergeObjects[i].left, afterMergeObjects[i].top, width, height);
		targetRect.push_back(rect);
	}

	return targetRect;
}

inline int Util::Sum(std::vector<int>& valueVec)
{
	auto result = 0;
	for (auto val : valueVec)
		result += val;

	return result;
}

inline void Util::DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex)
{
	std::stack<cv::Point> deepTrace;
	bitMap.at<int32_t>(r, c) = currentIndex;
	deepTrace.push(cv::Point(c, r));

	while (!deepTrace.empty())
	{
		auto curPos = deepTrace.top();
		deepTrace.pop();

		auto curR = curPos.y;
		auto curC = curPos.x;

		// up
		if (curR - 1 >= 0 && binaryFrame.at<uchar>(curR - 1, curC) == 0 && bitMap.at<int32_t>(curR - 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR - 1));
		}
		// down
		if (curR + 1 < binaryFrame.rows && binaryFrame.at<uchar>(curR + 1, curC) == 0 && bitMap.at<int32_t>(curR + 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR + 1));
		}
		// left
		if (curC - 1 >= 0 && binaryFrame.at<uchar>(curR, curC - 1) == 0 && bitMap.at<int32_t>(curR, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR));
		}
		// right
		if (curC + 1 < binaryFrame.cols && binaryFrame.at<uchar>(curR, curC + 1) == 0 && bitMap.at<int32_t>(curR, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR));
		}

		// up and left
		if (curR - 1 >= 0 && curC - 1 >= 0 && binaryFrame.at<uchar>(curR - 1, curC - 1) == 0 && bitMap.at<int32_t>(curR - 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR - 1));
		}
		// down and right
		if (curR + 1 < binaryFrame.rows && curC + 1 < binaryFrame.cols && binaryFrame.at<uchar>(curR + 1, curC + 1) == 0 && bitMap.at<int32_t>(curR + 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR + 1));
		}
		// left and down
		if (curC - 1 >= 0 && curR + 1 < binaryFrame.rows && binaryFrame.at<uchar>(curR + 1, curC - 1) == 0 && bitMap.at<int32_t>(curR + 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR + 1));
		}
		// right and up
		if (curC + 1 < binaryFrame.cols && curR - 1 >= 0 && binaryFrame.at<uchar>(curR - 1, curC + 1) == 0 && bitMap.at<int32_t>(curR - 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR - 1));
		}
	}
}

inline void Util::DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, uchar value)
{
	std::stack<cv::Point> deepTrace;
	bitMap.at<int32_t>(r, c) = currentIndex;
	deepTrace.push(cv::Point(c, r));

	while (!deepTrace.empty())
	{
		auto curPos = deepTrace.top();
		deepTrace.pop();

		auto curR = curPos.y;
		auto curC = curPos.x;

		// up
		if (curR - 1 >= 0 && binaryFrame.at<uchar>(curR - 1, curC) == value && bitMap.at<int32_t>(curR - 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR - 1));
		}
		// down
		if (curR + 1 < binaryFrame.rows && binaryFrame.at<uchar>(curR + 1, curC) == value && bitMap.at<int32_t>(curR + 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR + 1));
		}
		// left
		if (curC - 1 >= 0 && binaryFrame.at<uchar>(curR, curC - 1) == value && bitMap.at<int32_t>(curR, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR));
		}
		// right
		if (curC + 1 < binaryFrame.cols && binaryFrame.at<uchar>(curR, curC + 1) == value && bitMap.at<int32_t>(curR, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR));
		}
	}
}

inline void Util::DeepFirstSearch(const cv::Mat& grayFrame, cv::Mat& bitMap, int r, int c, int currentIndex)
{
	if (grayFrame.at<uchar>(r, c) == 0 && bitMap.at<int32_t>(r, c) == -1)
	{
		// center
		bitMap.at<int32_t>(r, c) = currentIndex;

		// up
		if (r - 1 >= 0)
			DeepFirstSearch(grayFrame, bitMap, r - 1, c, currentIndex);

		// down
		if (r + 1 < grayFrame.rows)
			DeepFirstSearch(grayFrame, bitMap, r + 1, c, currentIndex);

		// left
		if (c - 1 >= 0)
			DeepFirstSearch(grayFrame, bitMap, r, c - 1, currentIndex);

		// right
		if (c + 1 < grayFrame.cols)
			DeepFirstSearch(grayFrame, bitMap, r, c + 1, currentIndex);
	}
}
