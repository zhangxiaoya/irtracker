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

template <typename DataType>
class Util
{
public:

	static void BinaryMat(cv::Mat& mat);

	static double MeanMat(const cv::Mat& mat);

	static void ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects);

	static void FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType, DataType value = 0);

	static void GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject);

	static void ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, std::string title = "All Object");

	static void ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, DataType valueThreshHold = 0);

	static void ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<cv::Rect>& rects);

	static void ShowImage(cv::Mat curFrame);

	static DataType MaxOfVector(const std::vector<DataType>& data, const int& beginPos, const int& length);

	static DataType MinOfVector(const std::vector<DataType>& data, const int& beginPos, const int& length);

	static bool CompareUchar(uchar left, uchar right);

	static bool CompareConfidenceValue(ConfidenceElem left, ConfidenceElem right);

	static bool CompareTracker(TargetTracker left, TargetTracker right);

	static DataType AverageValue(const cv::Mat& curFrame, const cv::Rect& object);

	static std::vector<cv::Rect> GetCandidateTargets(const std::vector<FourLimits>& afterMergeObjects);

	static int Sum(const std::vector<int>& valueVec);

	static std::vector<DataType> ToFeatureVector(const cv::Mat& mat);

	static int FeatureDiff(const std::vector<DataType>& featureOne, const std::vector<DataType>& featureTwo);

	static DataType GetMinValueOfBlock(const cv::Mat& cuFrame);

	static DataType GetMaxValueOfBlock(const cv::Mat& mat);

	static DataType CalculateAverageValue(const cv::Mat& frame, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

	static DataType CalculateAverageValueWithBlockIndex(const cv::Mat& img, int blockX, int blockY);

	static void CalculateThreshHold(const cv::Mat& frame, DataType& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY);

	static void CalCulateCenterValue(const cv::Mat& frame, DataType& centerValue, const cv::Rect& rect);

	static double MaxOfConstLengthList(const double* data, const int& len);

	static double MinOfConstLengthList(const double* data, const int& len);

private:

	static void DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, DataType value = 0);

	static void DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, DataType value = 0);

	static void DeepFirstSearch(const cv::Mat& grayFrame, cv::Mat& bitMap, int r, int c, int currentIndex) = delete;
};

template <typename DataType>
void Util<DataType>::BinaryMat(cv::Mat& mat)
{
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			mat.at<DataType>(r, c) = mat.at<DataType>(r, c) > THRESHOLD ? 1 : 0;
}

template <typename DataType>
double Util<DataType>::MeanMat(const cv::Mat& mat)
{
	double sum = 0;
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			sum += static_cast<int>(mat.at<DataType>(r, c));

	return sum / (mat.rows * mat.cols);
}

template <typename DataType>
void Util<DataType>::ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects)
{
	cv::Mat colorFrame;
	cvtColor(grayFrame, colorFrame, CV_GRAY2RGB);

	for (auto i = 0; i < candidate_rects.size(); ++i)
		rectangle(colorFrame, candidate_rects[i], COLOR_RED);

	imshow("Color Frame", colorFrame);
}

template <typename DataType>
void Util<DataType>::FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType, DataType value)
{
	if (fieldType == FieldType::Eight)
		DFSWithoutRecursionEightField(binaryFrame, bitMap, r, c, currentIndex, value);
	else if (fieldType == FieldType::Four)
		DFSWithoutRecursionFourField(binaryFrame, bitMap, r, c, currentIndex, value);
	else
		std::cout << "FieldType Error!" << std::endl;
}

template <typename DataType>
void Util<DataType>::GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject)
{
	// top
	for (auto r = 0; r < bitMap.rows; ++r)
	{
		auto ptr = bitMap.ptr<int32_t>(r);
		for (auto c = 0; c < bitMap.cols; ++c)
		{
			auto curIndex = ptr[c];
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
		auto ptr = bitMap.ptr<int32_t>(r);
		for (auto c = 0; c < bitMap.cols; ++c)
		{
			auto curIndex = ptr[c];
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

template <typename DataType>
void Util<DataType>::ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, std::string title)
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

template <typename DataType>
void Util<DataType>::ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject, DataType valueThreshHold)
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

			if (curFrame.at<DataType>(allObject[i].top + 1, allObject[i].left + 1) < valueThreshHold)
				continue;
		}
		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);

		rectangle(colorFrame, rect, COLOR_GREEN);
	}

	imshow("All Candidate Targets", colorFrame);
}

template <typename DataType>
void Util<DataType>::ShowAllCandidateTargets(const cv::Mat& curFrame, const std::vector<cv::Rect>& rects)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for (auto i = 0; i < rects.size(); ++i)
	{
		rectangle(colorFrame, rects[i], COLOR_BLUE);
	}

	imshow("All Candidate Objects", colorFrame);
}

template <typename DataType>
void Util<DataType>::ShowImage(cv::Mat curFrame)
{
	imshow("Current Frame", curFrame);
	cv::waitKey(SHOW_DELAY);
}

template <typename DataType>
DataType Util<DataType>::MaxOfVector(const std::vector<DataType>& data, const int& beginPos, const int& length)
{
	auto maxResult = data[beginPos];
	for (auto i = beginPos; i < beginPos + length; ++i)
	{
		if (maxResult < data[i])
			maxResult = data[i];
	}
	return maxResult;
}

template <typename DataType>
DataType Util<DataType>::MinOfVector(const std::vector<DataType>& data, const int& beginPos, const int& length)
{
	auto minResult = data[beginPos];
	for (auto i = beginPos; i < beginPos + length; ++i)
	{
		if (minResult > data[i])
			minResult = data[i];
	}
	return minResult;
}

template <typename DataType>
bool Util<DataType>::CompareUchar(uchar left, uchar right)
{
	return left > right;
}

template <typename DataType>
bool Util<DataType>::CompareConfidenceValue(ConfidenceElem left, ConfidenceElem right)
{
	return left.confidenceVal > right.confidenceVal;
}

template <typename DataType>
bool Util<DataType>::CompareTracker(TargetTracker left, TargetTracker right)
{
	return left.timeLeft > right.timeLeft;
}

template <typename DataType>
DataType Util<DataType>::AverageValue(const cv::Mat& curFrame, const cv::Rect& rect)
{
	auto sumAll = 0;
	for (auto r = rect.y; r < rect.y + rect.height; ++r)
	{
		auto sumRow = 0;
		auto ptr = curFrame.ptr<DataType>(r);
		for (auto c = rect.x; c < rect.x + rect.width; ++c)
		{
			sumRow += ptr[c];
		}
		sumAll += (sumRow / rect.width);
	}

	return static_cast<DataType>(sumAll / rect.height);
}

template <typename DataType>
std::vector<cv::Rect> Util<DataType>::GetCandidateTargets(const std::vector<FourLimits>& afterMergeObjects)
{
	std::vector<cv::Rect> targetRect;

	for (auto i = 0; i < afterMergeObjects.size(); ++i)
	{
		auto object = afterMergeObjects[i];

		auto width = object.right - object.left + 1;
		auto height = object.bottom - object.top + 1;

		if (width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}

		targetRect.push_back(cv::Rect(object.left, object.top, width, height));
	}
	return targetRect;
}

template <typename DataType>
int Util<DataType>::Sum(const std::vector<int>& valueVec)
{
	auto result = 0;
	for (auto val : valueVec)
		result += val;

	return result;
}

template <typename DataType>
std::vector<DataType> Util<DataType>::ToFeatureVector(const cv::Mat& mat)
{
	std::vector<DataType> result(mat.cols * mat.rows, 0);

	auto index = 0;
	for (auto r = 0; r < mat.rows; ++r)
	{
		for (auto c = 0; c < mat.cols; ++c)
		{
			result[index++] = mat.at<DataType>(r, c) / 50;
		}
	}
	return result;
}

template <typename DataType>
int Util<DataType>::FeatureDiff(const std::vector<DataType>& featureOne, const std::vector<DataType>& featureTwo)
{
	auto sum = 0;
	for (auto i = 0; i < featureOne.size(); ++i)
	{
		sum += (featureOne[i] - featureTwo[i]) * (featureOne[i] - featureTwo[i]);
	}
	return sum;
}

template <typename DataType>
DataType Util<DataType>::GetMinValueOfBlock(const cv::Mat& mat)
{
	DataType minVal = (1 << (sizeof(DataType)) * 8) - 1;;
	for (auto r = 0; r < mat.rows; ++r)
	{
		auto ptr = mat.ptr<DataType>(r);
		for (auto c = 0; c < mat.cols; ++c)
		{
			if (minVal > ptr[c])
				minVal = ptr[c];
		}
	}
	return minVal;
}

template <typename DataType>
DataType Util<DataType>::GetMaxValueOfBlock(const cv::Mat& mat)
{
	DataType maxVal = 0;
	for (auto r = 0; r < mat.rows; ++r)
	{
		auto ptr = mat.ptr<DataType>(r);
		for (auto c = 0; c < mat.cols; ++c)
		{
			if (maxVal < ptr[c])
				maxVal = ptr[c];
		}
	}
	return maxVal;
}

template <typename DataType>
DataType Util<DataType>::CalculateAverageValue(const cv::Mat& frame, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	auto sumAll = 0;
	for (auto r = leftTopY; r < rightBottomY; ++r)
	{
		auto sumRow = 0;
		auto ptr = frame.ptr<DataType>(r);
		for (auto c = leftTopX; c < rightBottomX; ++c)
		{
			sumRow += ptr[c];
		}
		sumAll += sumRow / (rightBottomX - leftTopX);
	}

	return static_cast<DataType>(sumAll / (rightBottomY - leftTopY));
}

template <typename DataType>
DataType Util<DataType>::CalculateAverageValueWithBlockIndex(const cv::Mat& img, int blockX, int blockY)
{
	auto leftTopX = blockX * BLOCK_SIZE;
	auto leftTopY = blockY * BLOCK_SIZE;
	auto rightBottomX = leftTopX + BLOCK_SIZE >= IMAGE_WIDTH ? IMAGE_WIDTH : leftTopX + BLOCK_SIZE;
	auto rightBottomY = leftTopY + BLOCK_SIZE >= IMAGE_HEIGHT ? IMAGE_HEIGHT : leftTopY + BLOCK_SIZE;

	auto sumAll = 0;
	for (auto r = leftTopY; r < rightBottomY; ++r)
	{
		auto sumRow = 0;
		auto ptr = img.ptr<DataType>(r);
		for (auto c = leftTopX; c < rightBottomX; ++c)
		{
			sumRow += ptr[c];
		}
		sumAll += (sumRow / (rightBottomX - leftTopX));
	}

	return static_cast<DataType>(sumAll / (rightBottomY - leftTopY));
}

template <typename DataType>
void Util<DataType>::CalculateThreshHold(const cv::Mat& frame, DataType& threshHold, int leftTopX, int leftTopY, int rightBottomX, int rightBottomY)
{
	threshHold = CalculateAverageValue(frame, leftTopX, leftTopY, rightBottomX, rightBottomY);
	//threshHold += threshHold / 4;
}

template <typename DataType>
void Util<DataType>::CalCulateCenterValue(const cv::Mat& frame, DataType& centerValue, const cv::Rect& rect)
{
	auto centerX = rect.x + rect.width / 2;
	auto centerY = rect.y + rect.height / 2;

	auto sumAll = 0;
	sumAll += static_cast<int>(frame.at<DataType>(centerY, centerX));
	sumAll += static_cast<int>(frame.at<DataType>(centerY, centerX - 1));
	sumAll += static_cast<int>(frame.at<DataType>(centerY - 1, centerX));
	sumAll += static_cast<int>(frame.at<DataType>(centerY - 1, centerX - 1));
	centerValue = static_cast<DataType>(sumAll / 4);
}

template <typename DataType>
double Util<DataType>::MaxOfConstLengthList(const double* data, const int& len)
{
	auto MaxValue = data[0];
	for (auto i = 0; i < len; ++i)
	{
		if (data[i] - MaxValue > MinDiff)
			MaxValue = data[i];
	}
	return MaxValue;
}

template <typename DataType>
double Util<DataType>::MinOfConstLengthList(const double* data, const int& len)
{
	auto MinValue = data[0];
	for (auto i = 0; i < len; ++i)
	{
		if (MinValue - data[i] > MinDiff)
			MinValue = data[i];
	}
	return MinValue;
}

template <typename DataType>
void Util<DataType>::DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, DataType value)
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
		if (curR - 1 >= 0 && binaryFrame.at<DataType>(curR - 1, curC) == value && bitMap.at<int32_t>(curR - 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR - 1));
		}
		// down
		if (curR + 1 < binaryFrame.rows && binaryFrame.at<DataType>(curR + 1, curC) == value && bitMap.at<int32_t>(curR + 1, curC) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC) = currentIndex;
			deepTrace.push(cv::Point(curC, curR + 1));
		}
		// left
		if (curC - 1 >= 0 && binaryFrame.at<DataType>(curR, curC - 1) == value && bitMap.at<int32_t>(curR, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR));
		}
		// right
		if (curC + 1 < binaryFrame.cols && binaryFrame.at<DataType>(curR, curC + 1) == value && bitMap.at<int32_t>(curR, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR));
		}

		// up and left
		if (curR - 1 >= 0 && curC - 1 >= 0 && binaryFrame.at<DataType>(curR - 1, curC - 1) == value && bitMap.at<int32_t>(curR - 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR - 1));
		}
		// down and right
		if (curR + 1 < binaryFrame.rows && curC + 1 < binaryFrame.cols && binaryFrame.at<DataType>(curR + 1, curC + 1) == value && bitMap.at<int32_t>(curR + 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR + 1));
		}
		// left and down
		if (curC - 1 >= 0 && curR + 1 < binaryFrame.rows && binaryFrame.at<DataType>(curR + 1, curC - 1) == value && bitMap.at<int32_t>(curR + 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR + 1));
		}
		// right and up
		if (curC + 1 < binaryFrame.cols && curR - 1 >= 0 && binaryFrame.at<DataType>(curR - 1, curC + 1) == value && bitMap.at<int32_t>(curR - 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR - 1));
		}
	}
}

template <typename DataType>
void Util<DataType>::DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, DataType value)
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
		if (curR - 1 >= 0)
		{
			auto frameRowPtr = binaryFrame.ptr<DataType>(curR - 1);
			auto maskRowPtr = bitMap.ptr<int>(curR - 1);
			if (frameRowPtr[curC] == value && maskRowPtr[curC] == -1)
			{
				maskRowPtr[curC] = currentIndex;
				deepTrace.push(cv::Point(curC, curR - 1));
			}
		}
		// down
		if (curR + 1 < binaryFrame.rows)
		{
			auto frameRowPtr = binaryFrame.ptr<DataType>(curR + 1);
			auto maskRowPtr = bitMap.ptr<int>(curR + 1);
			if (frameRowPtr[curC] == value && maskRowPtr[curC] == -1)
			{
				maskRowPtr[curC] = currentIndex;
				deepTrace.push(cv::Point(curC, curR + 1));
			}
		}
		// left
		if (curC - 1 >= 0)
		{
			auto frameRowPtr = binaryFrame.ptr<DataType>(curR);
			auto maskRowPtr = bitMap.ptr<int>(curR);
			if (frameRowPtr[curC - 1] == value && maskRowPtr[curC - 1] == -1)
			{
				maskRowPtr[curC - 1] = currentIndex;
				deepTrace.push(cv::Point(curC - 1, curR));
			}
		}
		// right
		if (curC + 1 < binaryFrame.cols)
		{
			auto frameRowPtr = binaryFrame.ptr<DataType>(curR);
			auto maskRowPtr = bitMap.ptr<int>(curR);
			if (frameRowPtr[curC + 1] == value && maskRowPtr[curC + 1] == -1)
			{
				maskRowPtr[curC + 1] = currentIndex;
				deepTrace.push(cv::Point(curC + 1, curR));
			}
		}
	}
}
