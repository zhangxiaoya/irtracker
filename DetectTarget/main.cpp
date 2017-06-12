#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "DetectByMaxFilterAndAdptiveThreshHold.hpp"
#include "ConfidenceElem.hpp"
#include "SpecialUtil.hpp"

const auto SHOW_DELAY = 1;
const auto STEP = 10;

void InitVideoReader(cv::VideoCapture& video_capture)
{
	const char* firstImageList;

	if (AFTER_MAX_FILTER)
	{
		//		firstImageList = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_1_8bit_maxFilter_discrezated\\Frame_%04d.png";
		//		firstImageList = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";
		//				firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit_maxFilter_discrezated\\Frame_%04d.png";
		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";
	}
	else
	{
		// firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";
		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit\\Frame_%04d.png";
	}

	video_capture.open(firstImageList);
}

void LostMemory(double countX, double countY, int queueSize, int& currentIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap)
{
	currentIndex ++;
	currentIndex %= queueSize;
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			confidenceMap[y][x][currentIndex] = 0;
		}
	}
}

bool CheckIfInTopCount(const cv::Rect& rect, int searchIndex, const std::vector<ConfidenceElem>& confidenceElems)
{
	auto x = (rect.x + rect.width / 2) / STEP;
	auto y = (rect.y + rect.height / 2) / STEP;

	for (auto i = 0; i < searchIndex;++i)
	{
		if (confidenceElems[i].x == x && confidenceElems[i].y == y && confidenceElems[i].confidenceVal >= 40)
			return true;
	}
	return false;
}

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture;
	InitVideoReader(video_capture);

	cv::Mat curFrame;
	cv::Mat colorFrame;

	auto frameIndex = 0;

	char* fileNameFormat = ".\\ir_file_20170531_1000m_1\\Frame_%04d.png";
	const auto bufferSize = 100;
	char fileName[bufferSize];

	
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

				sort(allConfidence.begin(), allConfidence.end(), Util::ConfidenceCompare);

				const auto topCount = 5;
				auto searchIndex = 0;
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

				for (auto i = 0; i < targetRects.size(); ++i)
				{
					auto rect = targetRects[i];
					if (CheckIfInTopCount(rect, searchIndex, allConfidence))
						rectangle(colorFrame, cv::Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2), cv::Scalar(255, 255, 0));
				}

				LostMemory(countX, countY, queueSize, queueEndIndex, confidenceMap);

				imshow("last result", colorFrame);
				sprintf_s(fileName, bufferSize, fileNameFormat, frameIndex);
				imwrite(fileName, colorFrame);

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
