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
#include "TargetTracker.hpp"

const auto SHOW_DELAY = 1;
const auto TopCount = 5;
const auto WRITE_FILE_NAME_BUFFER_SIZE = 100;

const auto countX = ceil(static_cast<double>(320) / STEP);
const auto countY = ceil(static_cast<double>(256) / STEP);

void UpdateConfidenceMap(int queueEndIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap, const std::vector<cv::Rect>& targetRects)
{
	std::vector<std::vector<bool>> updateFlag(countY, std::vector<bool>(countX, false));
	for (auto i = 0; i < targetRects.size(); ++i)
	{
		auto rect = targetRects[i];
		auto x = (rect.x + rect.width / 2) / STEP;
		auto y = (rect.y + rect.height / 2) / STEP;

		if (updateFlag[y][x])
			continue;
		// center
		confidenceMap[y][x][queueEndIndex] += 20;
		updateFlag[y][x] = true;
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
			++currentTop;
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

void DrawRectangleForAllCandidateTargets(cv::Mat& colorFrame, const std::vector<ConfidenceElem>& allConfidence, const std::vector<cv::Rect>& targetRects, const int searchIndex, std::vector<std::vector<int>>& confidenceValueMap)
{
	std::vector<std::vector<bool>> updateFlag(countY, std::vector<bool>(countX, false));

	for (auto i = 0; i < targetRects.size(); ++i)
	{
		auto rect = targetRects[i];
		if (ConfidenceMapUtil::CheckIfInTopCount(rect, searchIndex, allConfidence))
		{
			rectangle(colorFrame, cv::Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2), BLUECOLOR);

			auto x = (rect.x + rect.width / 2) / STEP;
			auto y = (rect.y + rect.height / 2) / STEP;

			if (updateFlag[y][x])
				continue;
			confidenceValueMap[y][x] += 5;
			updateFlag[y][x] = true;
			// neighbor effect
			if (x - 1 >= 0)
				confidenceValueMap[y][x - 1] += 4;
			if (x + 1 < countX)
				confidenceValueMap[y][x + 1] += 4;
			if (y - 1 >= 0)
				confidenceValueMap[y - 1][x] += 4;
			if (y + 1 < countY)
				confidenceValueMap[y + 1][x] += 4;
		}
	}
}

void WriteLastResultToDisk(const cv::Mat& colorFrame, const int frameIndex, char writeFileName[])
{
	sprintf_s(writeFileName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalWriteFileNameFormat, frameIndex);
	imwrite(writeFileName, colorFrame);
}

int MaxNeighbor(const std::vector<std::vector<int>>& confidenceValueMap, int y, int x)
{
	auto maxResult = 0;
	if (x - 1 >= 0 && confidenceValueMap[y][x - 1] > maxResult)
		maxResult = confidenceValueMap[y][x - 1];
	if (x + 1 < countX && confidenceValueMap[y][x + 1] > maxResult)
		maxResult = confidenceValueMap[y][x + 1];
	if (y - 1 >= 0 && confidenceValueMap[y - 1][x] > maxResult)
		maxResult = confidenceValueMap[y - 1][x];
	if (y + 1 < countY && confidenceValueMap[y + 1][x] > maxResult)
		maxResult = confidenceValueMap[y + 1][x];
	return maxResult;
}

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture;

	InitVideoReader(video_capture);

	cv::Mat curFrame;
	cv::Mat colorFrame;

	auto frameIndex = 0;

	char writeFileName[WRITE_FILE_NAME_BUFFER_SIZE];

	const auto queueSize = 4;
	auto queueEndIndex = 0;

	std::vector<std::vector<std::vector<int>>> confidenceQueueMap(countY, std::vector<std::vector<int>>(countX, std::vector<int>(queueSize, 0)));
	std::vector<std::vector<int>> confidenceValueMap(countY, std::vector<int>(countX, 0));

	std::vector<ConfidenceElem> allConfidenceQueue(countX * countY);
	std::vector<ConfidenceElem> allConfidenceValues(countX * countY);

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

				UpdateConfidenceMap(queueEndIndex, confidenceQueueMap, targetRects);

				UpdateConfidenceVector(countX, countY, confidenceQueueMap, allConfidenceQueue);

				sort(allConfidenceQueue.begin(), allConfidenceQueue.end(), Util::ConfidenceCompare);

				auto searchIndex = 0;

				GetMostLiklyTargetsRect(allConfidenceQueue, TopCount, searchIndex);

				DrawRectangleForAllCandidateTargets(colorFrame, allConfidenceQueue, targetRects, searchIndex, confidenceValueMap);

				std::cout << "Before Draw Rect" <<std::endl;
				for (auto y = 0; y < countY; ++y)
				{
					for (auto x = 0; x < countX; ++x)
						std::cout << std::setw(2) << confidenceValueMap[y][x] << " ";

					std::cout << std::endl;
				}

				if(frameIndex > 6)
				{
					const auto maxTargetCount = 2;
					auto currentTargetCountIndex = 0;
					std::vector<cv::Point> blocksContainTargets;

					auto index = 0;
					for (auto x = 0; x < countX; ++x)
					{
						for (auto y = 0; y < countY; ++y)
						{
							allConfidenceValues[index].x = x;
							allConfidenceValues[index].y = y;
							allConfidenceValues[index++].confidenceVal = confidenceValueMap[y][x];
						}
					}
					
					sort(allConfidenceValues.begin(), allConfidenceValues.end(), Util::ConfidenceCompare);

					std::cout << "x = " << allConfidenceValues[0].x << " y = " << allConfidenceValues[0].y << " max Value = " << allConfidenceValues[0].confidenceVal << std::endl;

					for(auto i =0;i<allConfidenceValues.size();++i)
					{
						if(i == 0)
						{
							blocksContainTargets.push_back(cv::Point(allConfidenceValues[i].x, allConfidenceValues[i].y));
							currentTargetCountIndex++;
							continue;
						}
						if(allConfidenceValues[i].confidenceVal > 5 && allConfidenceValues[i].confidenceVal == allConfidenceValues[i-1].confidenceVal)
						{
							blocksContainTargets.push_back(cv::Point(allConfidenceValues[i].x, allConfidenceValues[i].y));
						}
						else
						{
							if(allConfidenceValues[i].confidenceVal >= 5)
							{
								blocksContainTargets.push_back(cv::Point(allConfidenceValues[i].x, allConfidenceValues[i].y));
								currentTargetCountIndex++;
								if (currentTargetCountIndex >= maxTargetCount)
									break;
							}
						}
					}

					for (auto i = 0; i < blocksContainTargets.size(); ++i)
					{
						auto findTargetFlag = false;

						for (auto j = 0; j < targetRects.size(); ++j)
						{
							auto rect = targetRects[j];
							auto x = (rect.x + rect.width / 2) / STEP;
							auto y = (rect.y + rect.height / 2) / STEP;

							if(x == blocksContainTargets[i].x && y == blocksContainTargets[i].y)
							{
								rectangle(colorFrame, cv::Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2), REDCOLOR);

								findTargetFlag = true;
							}
							// check neighbor
							else if (
								(x - 1 >= 0 && x - 1 == blocksContainTargets[i].x && y == blocksContainTargets[i].y) ||
								(y - 1 >= 0 && x == blocksContainTargets[i].x && y - 1 == blocksContainTargets[i].y) ||
								(x + 1 < countX && x + 1 == blocksContainTargets[i].x && y == blocksContainTargets[i].y) ||
								(y + 1 < countY && x == blocksContainTargets[i].x && y + 1 == blocksContainTargets[i].y))
							{
								confidenceValueMap[blocksContainTargets[i].y][blocksContainTargets[i].x] = MaxNeighbor(confidenceValueMap, y, x);

								rectangle(colorFrame, cv::Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2), REDCOLOR);

								findTargetFlag = true;
							}
						}

						if(!findTargetFlag)
						{
							confidenceValueMap[blocksContainTargets[i].y][blocksContainTargets[i].x] /= 2;
						}
					}

					std::cout << "After Draw Rect" << std::endl;
					for (auto y = 0; y < countY; ++y)
					{
						for (auto x = 0; x < countX; ++x)
							std::cout << std::setw(2) << confidenceValueMap[y][x] << " ";

						std::cout << std::endl;
					}

					for (auto x = 0; x < countX; ++x)
					{
						for (auto y = 0; y < countY; ++y)
						{
							if (confidenceValueMap[y][x] > 0)
							{
								confidenceValueMap[y][x] -= 3;
								if(confidenceValueMap[y][x] < 0)
									confidenceValueMap[y][x] = 0;
							}
						}
					}
				}

				ConfidenceMapUtil::LostMemory(countX, countY, queueSize, queueEndIndex, confidenceQueueMap);

				imshow("last result", colorFrame);

				WriteLastResultToDisk(colorFrame, frameIndex, writeFileName);

				std::cout << "Index : " << std::setw(4) << frameIndex++ << std::endl;

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
