#pragma once

#include <stack>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../Models/FourLimits.hpp"
#include "../Models/FieldType.hpp"
#include "../Models/ConfidenceElem.hpp"
#include "GlobalInitialUtil.hpp"
#include "../Tracker/TargetTracker.hpp"

class TargetTracker;

class Util
{
public:

	static void BinaryMat(cv::Mat& mat);

	static double MeanMat(const cv::Mat& mat);

	static void ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects);

	template <typename DATA_TYPE>
	static void FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType, DATA_TYPE value = 0);

	static void GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject);

	static void ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, std::string title = "All Object");

	static void ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, uchar valueThreshHold = 0);

	static void ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<cv::Rect>& rects);

	static void ShowImage(cv::Mat curFrame);

	static uchar MaxOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end);

	static uchar MinOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end);

	static bool CompareConfidenceValue(ConfidenceElem left, ConfidenceElem right);

	static bool CompareTracker(TargetTracker left, TargetTracker right);

	template <typename DATA_TYPE>
	static DATA_TYPE AverageValue(const cv::Mat& curFrame, const cv::Rect& object);

	template <typename DATA_TYPE>
	static std::vector<cv::Rect> GetCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& afterMergeObjects);

	static int Sum(const std::vector<int>& valueVec);

	static std::vector<uchar> ToFeatureVector(const cv::Mat& mat);

	static int FeatureDiff(const std::vector<unsigned char>& featureOne, const std::vector<unsigned char>& featureTwo);

	template<typename DATA_TYPE>
	static DATA_TYPE GetMinValueOfBlock(const cv::Mat& cuFrame);

	template<typename DATA_TYPE>
	static DATA_TYPE GetMaxValueOfBlock(const cv::Mat& mat);

	template <typename DATA_TYPE>
	static DATA_TYPE CalculateAverageValue(const cv::Mat& frame, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

private:

	template <typename DATA_TYPE>
	static void DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex);

	template <typename DATA_TYPE>
	static void DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, DATA_TYPE value = 0);

	static void DeepFirstSearch(const cv::Mat& grayFrame, cv::Mat& bitMap, int r, int c, int currentIndex);

	template <typename DATA_TYPE>
	static inline void CalculateThreshHold(const cv::Mat& frame, DATA_TYPE& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);
};

inline void Util::BinaryMat(cv::Mat& mat)
{
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			mat.at<uchar>(r, c) = mat.at<uchar>(r, c) > THRESHOLD ? 1 : 0;
}

inline double Util::MeanMat(const cv::Mat& mat)
{
	double sum = 0;
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			sum += static_cast<int>(mat.at<uchar>(r, c));

	return sum / (mat.rows * mat.cols);
}

inline void Util::ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects)
{
	cv::Mat colorFrame;
	cvtColor(grayFrame, colorFrame, CV_GRAY2RGB);

	for (auto i = 0; i < candidate_rects.size(); ++i)
		rectangle(colorFrame, candidate_rects[i], COLOR_RED);

	imshow("Color Frame", colorFrame);
}

template <typename DATA_TYPE>
void Util::FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType, DATA_TYPE value)
{
	if (fieldType == FieldType::Eight)
		DFSWithoutRecursionEightField<DATA_TYPE>(binaryFrame, bitMap, r, c, currentIndex);
	else if (fieldType == FieldType::Four)
		DFSWithoutRecursionFourField<DATA_TYPE>(binaryFrame, bitMap, r, c, currentIndex, value);
	else
		std::cout << "FieldType Error!" << std::endl;
}

inline void Util::GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject)
{
	// top
	for (auto r = 0; r < bitMap.rows; ++r)
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
		for (auto c = 0; c < bitMap.cols; ++c)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].bottom == -1)
				allObject[curIndex].bottom = r;
		}
	}
	// left
	for (auto c = 0; c < bitMap.cols; ++c)
	{
		for (auto r = 0; r < bitMap.rows; ++r)
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

inline void Util::ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, std::string title)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for (auto i = 0; i < allObject.size(); ++i)
	{
		auto width = allObject[i].right - allObject[i].left + 1;
		auto height = allObject[i].bottom - allObject[i].top + 1;
		if (width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}
		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);
		rectangle(colorFrame, rect, COLOR_BLUE);
	}

	imshow(title, colorFrame);
}

inline void Util::ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, uchar valueThreshHold)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for (auto i = 0; i < allObject.size(); ++i)
	{
		auto width = allObject[i].right - allObject[i].left + 1;
		auto height = allObject[i].bottom - allObject[i].top + 1;

		if (valueThreshHold != 0)
		{
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
		}
		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);

		rectangle(colorFrame, rect, COLOR_GREEN);
	}

	imshow("All Candidate Targets", colorFrame);
}

inline void Util::ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<cv::Rect>& rects)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for (auto i = 0; i < rects.size(); ++i)
	{
		rectangle(colorFrame, rects[i], COLOR_BLUE);
	}

	imshow("All Candidate Objects", colorFrame);
}

inline void Util::ShowImage(cv::Mat curFrame)
{
	imshow("Current Frame", curFrame);
	cv::waitKey(SHOW_DELAY);
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

inline bool Util::CompareConfidenceValue(ConfidenceElem left, ConfidenceElem right)
{
	return left.confidenceVal > right.confidenceVal;
}

inline bool Util::CompareTracker(TargetTracker left, TargetTracker right)
{
	return left.timeLeft > right.timeLeft;
}

template <typename DATA_TYPE>
DATA_TYPE Util::AverageValue(const cv::Mat& curFrame, const cv::Rect& rect)
{
	auto sumAll = 0;
	for (auto r = rect.y; r < rect.y + rect.height; ++r)
	{
		auto sumRow = 0;
		for (auto c = rect.x; c < rect.x + rect.width; ++c)
			sumRow += curFrame.at<DATA_TYPE>(r, c);
		sumAll += (sumRow / rect.width);
	}

	return sumAll / rect.height;
}

template <typename DATA_TYPE>
std::vector<cv::Rect> Util::GetCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& afterMergeObjects)
{
	std::vector<cv::Rect> targetRect;

	for (auto i = 0; i < afterMergeObjects.size(); ++i)
	{
		DATA_TYPE threshHold = 0;

		auto object = afterMergeObjects[i];

		auto width = object.right - object.left + 1;
		auto height = object.bottom - object.top + 1;

		auto surroundBoxWidth = 2 * width;
		auto surroundBoxHeight = 2 * height;

		auto centerX = (object.right + object.left) / 2;
		auto centerY = (object.bottom + object.top) / 2;

		auto leftTopX = centerX - surroundBoxWidth / 2;
		if (leftTopX < 0)
			leftTopX = 0;

		auto leftTopY = centerY - surroundBoxHeight / 2;
		if (leftTopY < 0)
			leftTopY = 0;

		auto rightBottomX = leftTopX + surroundBoxWidth;
		if (rightBottomX > curFrame.cols)
			rightBottomX = curFrame.cols;

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY > curFrame.rows)
			rightBottomY = curFrame.rows;

		CalculateThreshHold<DATA_TYPE>(curFrame, threshHold, leftTopX, leftTopY, rightBottomX, rightBottomY);

		if (width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}

		if ((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
			(width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			continue;

		auto rect = cv::Rect(object.left, object.top, width, height);

		if (curFrame.at<DATA_TYPE>(centerY, centerX) < threshHold)
			continue;

		targetRect.push_back(rect);
	}
	return targetRect;
}

inline int Util::Sum(const std::vector<int>& valueVec)
{
	auto result = 0;
	for (auto val : valueVec)
		result += val;

	return result;
}

inline std::vector<uchar> Util::ToFeatureVector(const cv::Mat& mat)
{
	std::vector<uchar> result(mat.cols * mat.rows, 0);

	auto index = 0;
	for (auto r = 0; r < mat.rows; ++r)
	{
		for (auto c = 0; c < mat.cols; ++c)
		{
			result[index++] = mat.at<uchar>(r, c) / 50;
		}
	}
	return result;
}

inline int Util::FeatureDiff(const std::vector<unsigned char>& featureOne, const std::vector<unsigned char>& featureTwo)
{
	auto sum = 0;
	for (auto i = 0; i < featureOne.size(); ++i)
	{
		sum += (featureOne[i] - featureTwo[i]) * (featureOne[i] - featureTwo[i]);
	}
	return sum;
}

template<typename DATA_TYPE>
DATA_TYPE Util::GetMinValueOfBlock(const cv::Mat& mat)
{
	DATA_TYPE minVal = (1 << sizeof(DATA_TYPE)) - 1;
	for (auto r = 0; r < mat.rows; ++r)
	{
		for (auto c = 0; c < mat.cols; ++c)
		{
			if (minVal > mat.at<DATA_TYPE>(r, c))
				minVal = mat.at<DATA_TYPE>(r, c);
		}
	}
	return minVal;
}

template<typename DATA_TYPE>
DATA_TYPE Util::GetMaxValueOfBlock(const cv::Mat& mat)
{
	DATA_TYPE maxVal = 0;
	for (auto r = 0; r < mat.rows; ++r)
	{
		for (auto c = 0; c < mat.cols; ++c)
		{
			if (maxVal < mat.at<DATA_TYPE>(r, c))
				maxVal = mat.at<DATA_TYPE>(r, c);
		}
	}
	return maxVal;
}

template<typename DATA_TYPE>
DATA_TYPE Util::CalculateAverageValue(const cv::Mat& frame, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	DATA_TYPE sumAll = 0;
	for (auto r = leftTopY; r < rightBottomY; ++r)
	{
		auto sumRow = 0;
		for (auto c = leftTopX; c < rightBottomX; ++c)
		{
			sumRow += frame.at<DATA_TYPE>(r, c);
		}
		sumAll += (sumRow / (rightBottomX - leftTopX));
	}

	return sumAll / (rightBottomY - leftTopY);
}

template <typename DATA_TYPE>
void Util::DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex)
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
		if (curR - 1 >= 0 && binaryFrame.at<DATA_TYPE>(curR - 1, curC) == 0 && bitMap.at<int32_t>(curR - 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR - 1));
		}
		// down
		if (curR + 1 < binaryFrame.rows && binaryFrame.at<DATA_TYPE>(curR + 1, curC) == 0 && bitMap.at<int32_t>(curR + 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR + 1));
		}
		// left
		if (curC - 1 >= 0 && binaryFrame.at<DATA_TYPE>(curR, curC - 1) == 0 && bitMap.at<int32_t>(curR, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR));
		}
		// right
		if (curC + 1 < binaryFrame.cols && binaryFrame.at<DATA_TYPE>(curR, curC + 1) == 0 && bitMap.at<int32_t>(curR, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR));
		}

		// up and left
		if (curR - 1 >= 0 && curC - 1 >= 0 && binaryFrame.at<DATA_TYPE>(curR - 1, curC - 1) == 0 && bitMap.at<int32_t>(curR - 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR - 1));
		}
		// down and right
		if (curR + 1 < binaryFrame.rows && curC + 1 < binaryFrame.cols && binaryFrame.at<DATA_TYPE>(curR + 1, curC + 1) == 0 && bitMap.at<int32_t>(curR + 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR + 1));
		}
		// left and down
		if (curC - 1 >= 0 && curR + 1 < binaryFrame.rows && binaryFrame.at<DATA_TYPE>(curR + 1, curC - 1) == 0 && bitMap.at<int32_t>(curR + 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR + 1));
		}
		// right and up
		if (curC + 1 < binaryFrame.cols && curR - 1 >= 0 && binaryFrame.at<DATA_TYPE>(curR - 1, curC + 1) == 0 && bitMap.at<int32_t>(curR - 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR - 1));
		}
	}
}

template <typename DATA_TYPE>
void Util::DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, DATA_TYPE value)
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
		if (curR - 1 >= 0 && binaryFrame.at<DATA_TYPE>(curR - 1, curC) == value && bitMap.at<int32_t>(curR - 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR - 1));
		}
		// down
		if (curR + 1 < binaryFrame.rows && binaryFrame.at<DATA_TYPE>(curR + 1, curC) == value && bitMap.at<int32_t>(curR + 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR + 1));
		}
		// left
		if (curC - 1 >= 0 && binaryFrame.at<DATA_TYPE>(curR, curC - 1) == value && bitMap.at<int32_t>(curR, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR));
		}
		// right
		if (curC + 1 < binaryFrame.cols && binaryFrame.at<DATA_TYPE>(curR, curC + 1) == value && bitMap.at<int32_t>(curR, curC + 1) == -1)
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

template <typename DATA_TYPE>
void Util::CalculateThreshHold(const cv::Mat& frame, DATA_TYPE& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	threshHold = CalculateAverageValue<DATA_TYPE>(frame, leftTopX, leftTopY, rightBottomX, rightBottomY);
//	threshHold += threshHold / 4;
}
