#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "DetectByMaxFilterAndAdptiveThreshHold.hpp"
#include "ConfidenceElem.hpp"
#include "SpecialUtil.hpp"
#include "ConfidenceMapUtil.hpp"
#include "GlobalInitialUtil.hpp"

const auto SHOW_DELAY = 1;
const auto TopCount = 5;
const auto WRITE_FILE_NAME_BUFFER_SIZE = 100;

void UpdateConfidenceMap(int queueEndIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap, const std::vector<cv::Rect>& targetRects)
{
	for (auto i = 0; i < targetRects.size(); ++i)
	{
		auto rect = targetRects[i];
		auto x = (rect.x + rect.width / 2) / STEP;
		auto y = (rect.y + rect.height / 2) / STEP;

		// center
		confidenceMap[y][x][queueEndIndex] += 20;
		// up
		confidenceMap[rect.y / STEP][x][queueEndIndex] += 1;
		// down
		confidenceMap[(rect.y + rect.height - 1) / STEP][x][queueEndIndex] += 1;
		// left
		confidenceMap[y][rect.x / STEP][queueEndIndex] += 1;
		// right
		confidenceMap[y][(rect.x + rect.width - 1) / STEP][queueEndIndex] += 1;
	}
}

void UpdateConfidenceVector(double countX, double countY, const std::vector<std::vector<std::vector<int>>>& confidenceMap, std::vector<ConfidenceElem>& allConfidence)
{
	auto confidenceIndex = 0;
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			allConfidence[confidenceIndex].x = x;
			allConfidence[confidenceIndex].y = y;
			allConfidence[confidenceIndex++].confidenceVal = Util::Sum(confidenceMap[y][x]);
		}
	}
}

void GetMostLiklyTargetsRect(const std::vector<ConfidenceElem>& allConfidence, const int topCount, int& searchIndex)
{
	auto currentTop = 0;

	while (searchIndex < allConfidence.size())
	{
		if (searchIndex == 0)
		{
			++searchIndex;
			continue;
		}
		if (allConfidence[currentTop].confidenceVal == allConfidence[searchIndex].confidenceVal)
		{
			++searchIndex;
		}
		else
		{
			++currentTop;
			++searchIndex;
			if (currentTop >= topCount)
				break;
		}
	}
}

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture;

	InitVideoReader(video_capture);

	cv::Mat curFrame;
	cv::Mat colorFrame;

	auto frameIndex = 0;

	char writeFileNameFormat[] = ".\\ir_file_20170531_1000m_1\\Frame_%04d.png";
	char writeFileName[WRITE_FILE_NAME_BUFFER_SIZE];

	
	auto countX = ceil(static_cast<double>(320) / STEP);
	auto countY = ceil(static_cast<double>(256) / STEP);

	const auto queueSize = 4;
	auto queueEndIndex = 0;

	std::vector<std::vector<std::vector<int>>> confidenceMap(countY, std::vector<std::vector<int>>(countX, std::vector<int>(queueSize, 0)));
	std::vector<ConfidenceElem> allConfidence(countX * countY);

	if (video_capture.isOpened())
	{
		std::cout << "Open Image List Success!" << std::endl;

		while (!curFrame.empty() || frameIndex == 0)
		{
			video_capture >> curFrame;
			if (!curFrame.empty())
			{
				SpecialUtil::RemoveInvalidPixel(curFrame);

				imshow("Current Frame", curFrame);
				cv::waitKey(SHOW_DELAY);

				cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

				//				DetectByDiscontinuity::Detect(curFrame);

				//				DetectByBinaryBitMap::Detect(curFrame);

				//				DetectByMultiScaleLocalDifference::Detect(curFrame);

				auto targetRects = DetectByMaxFilterAndAdptiveThreshHold::Detect(curFrame);

				UpdateConfidenceMap(queueEndIndex, confidenceMap, targetRects);

				UpdateConfidenceVector(countX, countY, confidenceMap, allConfidence);

				sort(allConfidence.begin(), allConfidence.end(), Util::ConfidenceCompare);

				auto searchIndex = 0;

				GetMostLiklyTargetsRect(allConfidence, TopCount, searchIndex);

				for (auto i = 0; i < targetRects.size(); ++i)
				{
					auto rect = targetRects[i];
					if (ConfidenceMapUtil::CheckIfInTopCount(rect, searchIndex, allConfidence))
						rectangle(colorFrame, cv::Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2), cv::Scalar(255, 255, 0));
				}

				ConfidenceMapUtil::LostMemory(countX, countY, queueSize, queueEndIndex, confidenceMap);

				imshow("last result", colorFrame);
				sprintf_s(writeFileName, WRITE_FILE_NAME_BUFFER_SIZE, writeFileNameFormat, frameIndex);
				imwrite(writeFileName, colorFrame);

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
