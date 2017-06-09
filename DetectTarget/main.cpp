#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "FieldType.hpp"
#include "FourLimits.hpp"
#include "Util.hpp"
#include "DetectByDiscontinuity.hpp"
#include "DetectByBinaryBitMap.hpp"
#include "DetectByMultiscaleLocalDifference.hpp"

const auto DELAY = 10;

const auto CONTIUNITY_THRESHHOLD = 0.4;


//const char* firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\second\\frame_%04d.png";
//const char* firstImageList = "D:\\Bag\\Code_VS15\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";
const char* firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";

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

	return Util::MaxOfVector(pixelValues.begin(), pixelValues.end());
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
			Util::FindNeighbor(filtedFrame, blockMap, r, c, currentIndex++, FieldType::Four, val);
		}
	}
	return currentIndex;
}

void DetectByMaxFilterAndAdptiveThreshHold(cv::Mat curFrame)
{
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

	sort(allValues.begin(), allValues.end(), Util::comp);

	auto iterator = unique(allValues.begin(), allValues.end());
	allValues.resize(distance(allValues.begin(), iterator));

	for (auto i = 0; i < topCount; ++i)
		maxValues[i] = allValues[i];

	cv::Mat blockMap(cv::Size(filtedFrame.cols, filtedFrame.rows), CV_32SC1, cv::Scalar(-1));
	auto totalObject = GetBlocks(filtedFrame, blockMap);

	std::vector<FourLimits> allObjects(totalObject);
	Util::GetRectangleSize(blockMap, allObjects, totalObject);

	std::cout << "Max Value Threh Hold = " << static_cast<int>(maxValues[2]) <<std::endl;
	Util::ShowAllObject(curFrame, allObjects);
	Util::ShowCandidateTargets(curFrame, allObjects, maxValues[4]);
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

				DetectByBinaryBitMap::DetectTargetsByBitMap(curFrame);

				DetectByMultiScaleLocalDifference::MultiscaleLocalDifferenceContrast(curFrame);

				DetectByMaxFilterAndAdptiveThreshHold(curFrame);

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
