#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <stack>
#include <iterator>

#include "FieldType.hpp"
#include "FourLimits.hpp"
#include "Util.hpp"
#include "DetectByDiscontinuity.hpp"

const auto DELAY = 10;

const auto CONTIUNITY_THRESHHOLD = 0.4;

const auto TARGET_WIDTH_MIN_LIMIT = 2;
const auto TARGET_HEIGHT_MIN_LIMIT = 2;
const auto TARGET_WIDTH_MAX_LIMIT = 16;
const auto TARGET_HEIGHT_MAX_LIMIT = 16;

//const char* firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\second\\frame_%04d.png";
//const char* firstImageList = "D:\\Bag\\Code_VS15\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";
const char* firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";

inline bool comp(uchar left, uchar right)
{
	return left > right;
}

void DeepFirstSearch(const cv::Mat& grayFrame, cv::Mat& bitMap, int r, int c, int currentIndex)
{
	if(grayFrame.at<uchar>(r,c) == 0 && bitMap.at<int32_t>(r,c) == -1)
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
		if(c - 1 >= 0)
			DeepFirstSearch(grayFrame, bitMap, r, c-1, currentIndex);

		// right
		if (c + 1 < grayFrame.cols)
			DeepFirstSearch(grayFrame, bitMap, r, c + 1, currentIndex);
	}
}

void DFSWithoutRecursionFourField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, uchar value = 0)
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

void DFSWithoutRecursionEightField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex)
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
		if (curR + 1 < binaryFrame.rows && curC + 1 < binaryFrame.cols && binaryFrame.at<uchar>(curR + 1, curC+1) == 0 && bitMap.at<int32_t>(curR + 1, curC+1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC+1) = currentIndex;
			deepTrace.push(cv::Point(curC+1, curR + 1));
		}
		// left and down
		if (curC - 1 >= 0 && curR + 1 < binaryFrame.rows && binaryFrame.at<uchar>(curR + 1, curC - 1) == 0 && bitMap.at<int32_t>(curR + 1, curC - 1) == -1)
		{
			bitMap.at<int32_t>(curR + 1, curC - 1) = currentIndex;
			deepTrace.push(cv::Point(curC - 1, curR + 1));
		}
		// right and up
		if (curC + 1 < binaryFrame.cols && curR -1 >= 0 && binaryFrame.at<uchar>(curR - 1, curC + 1) == 0 && bitMap.at<int32_t>(curR - 1, curC + 1) == -1)
		{
			bitMap.at<int32_t>(curR - 1, curC + 1) = currentIndex;
			deepTrace.push(cv::Point(curC + 1, curR-1));
		}
	}
}

void FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType,uchar value = 0)
{
	if(fieldType == FieldType::Eight)
		DFSWithoutRecursionEightField(binaryFrame, bitMap, r, c, currentIndex);
	else if(fieldType == FieldType::Four)
		DFSWithoutRecursionFourField(binaryFrame, bitMap, r, c, currentIndex,value);
	else
		std::cout << "FieldType Error!" << std::endl;
}

int GetBitMap(const cv::Mat& binaryFrame, cv::Mat& bitMap)
{
	auto currentIndex = 0;
	for (auto r = 0; r < binaryFrame.rows; ++r)
	{
		for (auto c = 0; c < binaryFrame.cols; ++c)
		{
			if(binaryFrame.at<uchar>(r,c) == 1)
				continue;
			if(bitMap.at<int32_t>(r,c) != -1)
				continue;

			FindNeighbor(binaryFrame, bitMap, r, c, currentIndex++, FieldType::Eight);
		}
	}
	return currentIndex;
}

void GetRectangleSize(const cv::Mat& bitMap, std::vector<FourLimits>& allObject, int totalObject)
{
	// top
	for(auto r = 0;r<bitMap.rows;++r)
	{
		for(auto c =0;c < bitMap.cols;++c)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].top == -1)
				allObject[curIndex].top = r;
		}
	}
	// bottom
	for(auto r = bitMap.rows-1;r >= 0;--r)
	{
		for(auto c = 0;c< bitMap.cols;++c)
		{
			auto curIndex = bitMap.at<int32_t>(r, c);
			if (curIndex != -1 && allObject[curIndex].bottom == -1)
				allObject[curIndex].bottom = r;
		}
	}
	// left
	for(auto c = 0;c<bitMap.cols;++c)
	{
		for(auto r =0;r <bitMap.rows;++r)
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

void ShowAllObject(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject)
{
	cv::Mat colorFrame;
	cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

	for(auto i =0;i<allObject.size();++i)
	{
		auto width = allObject[i].right - allObject[i].left + 1;
		auto height = allObject[i].bottom - allObject[i].top + 1;
		if(width <= 0 || height <= 0)
		{
			std::cout << "Rect Error, and index is " << i << std::endl;
			continue;
		}
		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);
		rectangle(colorFrame, rect, BLUECOLOR);
	}

	imshow("All Object", colorFrame);
}

void ShowCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject,uchar valueThreshHold = 0)
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

		if((width < TARGET_WIDTH_MIN_LIMIT || height < TARGET_HEIGHT_MIN_LIMIT) ||
		   (width > TARGET_WIDTH_MAX_LIMIT || height > TARGET_HEIGHT_MAX_LIMIT))
			continue;

		if(curFrame.at<uchar>(allObject[i].top+1,allObject[i].left+1) < valueThreshHold)
			continue;

		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);
		rectangle(colorFrame, rect, GREENCOLOR);
	}

	imshow("Candidate Targets", colorFrame);
}

void DetectTargetsByBitMap(const cv::Mat& curFrame)
{
	cv::Mat binaryFrame;
	curFrame.copyTo(binaryFrame);
	Util::BinaryMat(binaryFrame);

	cv::Mat bitMap(cv::Size(binaryFrame.cols, binaryFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBitMap(binaryFrame, bitMap);

	std::vector<FourLimits> allObjects(totalObject);
	GetRectangleSize(bitMap,allObjects,totalObject);

	ShowAllObject(curFrame,allObjects);
	ShowCandidateTargets(curFrame, allObjects);
}

uint8_t GetAverageGrayValueOfKNeighbor(const cv::Mat& curFrame, int r, int c, int i)
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
		if(row >= 0 && row < curFrame.rows)
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

uchar MaxOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end)
{
	auto maxResult = *begin;
	for (auto it = begin;it != end;++it)
	{
		if (maxResult < *it)
			maxResult = *it;
	}
	return maxResult;
}

uchar MinOfVector(const std::vector<uchar>::iterator& begin, const std::vector<uchar>::iterator& end)
{
	auto minResult = *begin;
	for (auto it = begin; it != end; ++it)
	{
		if (minResult > *it)
			minResult = *it;
	}
	return minResult;
}

void MultiscaleLocalDifferenceContrast(cv::Mat curFrame)
{
	cv::Mat mldFilterFrame(cv::Size(curFrame.cols,curFrame.rows),CV_8UC1,cv::Scalar(0));

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

			auto maxVal = MaxOfVector(averageOfKNeighbor.begin(),averageOfKNeighbor.end());
			auto minVal = MinOfVector(averageOfKNeighbor.begin(), averageOfKNeighbor.end());

			auto squareDiff = (maxVal - minVal) * (maxVal - minVal);
						
			if (squareDiff == 0)
			{
				mldFilterFrame.at<uchar>(r, c) = maxVal;
				std::cout << "Dummy" <<std::endl;
				continue;
			}

			for (auto i = 0; i < L - 1; ++i)
			{
				contrastOfKNeighbor.push_back((averageOfKNeighbor[i] - averageOfKNeighbor[L - 1]) * (averageOfKNeighbor[i] - averageOfKNeighbor[L - 1]) / squareDiff);
			}

			contrastOfKNeighbor.push_back(0);

			mldFilterFrame.at<uchar>(r, c) = MaxOfVector(contrastOfKNeighbor.begin(), contrastOfKNeighbor.end());
		}
	}

	imshow("Map", mldFilterFrame);
}

unsigned char GetMaxPixelValue(const cv::Mat& curFrame, int r, int c, int kernelSize)
{
	auto radius = kernelSize / 2;
	auto leftTopX = c - radius;
	auto leftTopY = r - radius;

	auto rightBottomX = leftTopX + 2 * radius;
	auto rightBottomY = leftTopY + 2 * radius;

	std::vector<uchar> pixelValues;

	for (auto row = leftTopY; row <= rightBottomY; ++row)
	{
		if (row >= 0 && row < curFrame.rows)
		{
			for (auto col = leftTopX; col <= rightBottomX; ++col)
			{
				if (col >= 0 && col < curFrame.cols)
					pixelValues.push_back(curFrame.at<uchar>(row, col));
			}
		}
	}

	return MaxOfVector(pixelValues.begin(), pixelValues.end());
}

void MaxFilter(const cv::Mat& curFrame, cv::Mat& filtedFrame, int kernelSize)
{
	std::vector<uchar> pixelVector;

	for (auto r = 0; r < curFrame.rows; ++r)
	{
		for (auto c = 0; c < curFrame.cols; ++c)
		{
			pixelVector.clear();
			filtedFrame.at<uchar>(r,c) = GetMaxPixelValue(curFrame,r,c,kernelSize);
		}
	}
}

int GetBlocks(const cv::Mat& filtedFrame, cv::Mat& blockMap)
{
	auto currentIndex = 0;
	for (auto r = 0; r < filtedFrame.rows; ++r)
	{
		for (auto c = 0; c < filtedFrame.cols; ++c)
		{
			if (blockMap.at<int32_t>(r, c) != -1)
				continue;

			auto val = filtedFrame.at<uchar>(r, c);
			FindNeighbor(filtedFrame, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	return currentIndex;
}

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture;
	video_capture.open(firstImageList);

	cv::Mat curFrame;
	auto frameIndex = 0;

	if(video_capture.isOpened())
	{
		std::cout << "Open Image List Success!" << std::endl;

		while (!curFrame.empty() || frameIndex == 0)
		{
			video_capture >> curFrame;
			if(!curFrame.empty())
			{
				imshow("Current Frame", curFrame);
				cv::waitKey(DELAY);

				DetectByDiscontinuity::DetectTarget(curFrame);

//				DetectTargetsByBitMap(curFrame);

//				MultiscaleLocalDifferenceContrast(curFrame);

				cv::Mat filtedFrame(cv::Size(curFrame.cols,curFrame.rows),CV_8UC1);
				auto kernelSize = 3;

				MaxFilter(curFrame, filtedFrame, kernelSize);

				imshow("Max Filter", filtedFrame);

				const auto topCount = 5;
				std::vector<uchar> maxValues(topCount, 0);

				std::vector<uchar> allValues;

				for (auto r = 0; r < filtedFrame.rows; ++r)
					for (auto c = 0; c < filtedFrame.cols; ++c)
						allValues.push_back(filtedFrame.at<uchar>(r, c));

				sort(allValues.begin(), allValues.end(), comp);

				auto iterator = unique(allValues.begin(), allValues.end());
				allValues.resize(distance(allValues.begin(), iterator));

				for (auto i = 0; i < topCount; ++i)
					maxValues[i] = allValues[i];

				cv::Mat blockMap(cv::Size(filtedFrame.cols, filtedFrame.rows), CV_32SC1, cv::Scalar(-1));
				auto totalObject = GetBlocks(filtedFrame, blockMap);

				std::vector<FourLimits> allObjects(totalObject);
				GetRectangleSize(blockMap, allObjects, totalObject);

				std::cout << "Max Value Threh Hold = " << static_cast<int>(maxValues[2]) <<std::endl;
				ShowAllObject(curFrame, allObjects);
				ShowCandidateTargets(curFrame, allObjects, maxValues[4]);

				std::cout << "Index : " << std::setw(4) << frameIndex << std::endl;
				++frameIndex;
			}
		}

		cv::destroyAllWindows();
	}
	else
	{
		std::cout << "Open Image List Failed" << std::endl;
	}

	system("pause");
	return 0;
}
