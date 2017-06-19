#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "DetectByMaxFilterAndAdptiveThreshold.hpp"
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

void UpdateConfidenceMap(int queueEndIndex, std::vector<std::vector<std::vector<int>>>& confidenceMap, const std::vector<cv::Rect>& targetRects, FieldType fieldType = Four)
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
		confidenceMap[y][x][queueEndIndex] += 10;
		updateFlag[y][x] = true;
		// up
		confidenceMap[y - 1 >= 0 ? y - 1 : 0][x][queueEndIndex] += 1;
		// down
		confidenceMap[y + 1 < countY ? y + 1 : countY - 1][x][queueEndIndex] += 1;
		// left
		confidenceMap[y][x - 1 > 0 ? x - 1 : 0][queueEndIndex] += 1;
		// right
		confidenceMap[y][x + 1 < countX ? x + 1 : countX - 1][queueEndIndex] += 1;

		if (fieldType == Eight)
		{
			// up left
			confidenceMap[y - 1 >= 0 ? y - 1 : 0][x - 1 >= 0 ? x - 1 : 0][queueEndIndex] += 1;
			// down right
			confidenceMap[y + 1 < countY ? y + 1 : countY - 1][x + 1 < countX ? x + 1 : countX - 1][queueEndIndex] += 1;
			// left down
			confidenceMap[y + 1 < countY ? y + 1 : countY - 1][x - 1 > 0 ? x - 1 : 0][queueEndIndex] += 1;
			// right up
			confidenceMap[y - 1 >= 0 ? y - 1 : 0][x + 1 < countX ? x + 1 : countX - 1][queueEndIndex] += 1;
		}
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

	sort(allConfidence.begin(), allConfidence.end(), Util::CompareConfidenceValue);
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

void DrawRectangleForAllCandidateTargets(cv::Mat& colorFrame, const std::vector<ConfidenceElem>& allConfidence, std::vector<cv::Rect>& targetRects, const int searchIndex, std::vector<std::vector<int>>& confidenceValueMap)
{
	std::vector<std::vector<bool>> updateFlag(countY, std::vector<bool>(countX, false));

	for (auto it = targetRects.begin(); it != targetRects.end(); ++it)
	{
		auto rect = *it;
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
		else
		{
			it = targetRects.erase(it);
			if (it == targetRects.end())
				break;
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

int MinNeighbor(const std::vector<std::vector<int>>& confidenceValueMap, int y, int x)
{
	auto minResult = 10000;
	if (x - 1 >= 0 && confidenceValueMap[y][x - 1] < minResult)
		minResult = confidenceValueMap[y][x - 1];
	if (x + 1 < countX && confidenceValueMap[y][x + 1] < minResult)
		minResult = confidenceValueMap[y][x + 1];
	if (y - 1 >= 0 && confidenceValueMap[y - 1][x] < minResult)
		minResult = confidenceValueMap[y - 1][x];
	if (y + 1 < countY && confidenceValueMap[y + 1][x] < minResult)
		minResult = confidenceValueMap[y + 1][x];
	return minResult;
}

bool TrackerDecited(const cv::Rect& rect, int x, int y, int trackerIndex, const cv::Mat frame)
{
	if (trackerIndex != 0)
	{
		if (GlobalTrackerList[trackerIndex - 1].blockX != x)
			GlobalTrackerList[trackerIndex - 1].blockX = x;
		if (GlobalTrackerList[trackerIndex - 1].blockY != y)
			GlobalTrackerList[trackerIndex - 1].blockY = y;

		GlobalTrackerList[trackerIndex - 1].leftTopX = rect.x;
		GlobalTrackerList[trackerIndex - 1].leftTopY = rect.y;
		GlobalTrackerList[trackerIndex - 1].targetRect = rect;
		GlobalTrackerList[trackerIndex - 1].ExtendLifeTime();
		//		GlobalTrackerList[trackerIndex - 1].feature = Util::ToFeatureVector(frame(rect));
		return true;
	}
	return false;
}

void PrintConfidenceValueMap(const std::vector<std::vector<int>>& confidenceValueMap, char* text)
{
	std::cout << text << std::endl;
	for (auto y = 0; y < countY; ++y)
	{
		for (auto x = 0; x < countX; ++x)
			std::cout << std::setw(2) << confidenceValueMap[y][x] << " ";

		std::cout << std::endl;
	}
}

void UpdateConfidenceValueVector(const std::vector<std::vector<int>>& confidenceValueMap, std::vector<ConfidenceElem>& allConfidenceValues)
{
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

	sort(allConfidenceValues.begin(), allConfidenceValues.end(), Util::CompareConfidenceValue);

	std::cout << "x = " << allConfidenceValues[0].x << " y = " << allConfidenceValues[0].y << " max Value = " << allConfidenceValues[0].confidenceVal << std::endl;
}

void GetTopCountBlocksWhichContainsTargets(const std::vector<ConfidenceElem>& allConfidenceValues, const int maxTargetCount, std::vector<cv::Point>& blocksContainTargets)
{
	auto currentTargetCountIndex = 0;

	for (auto i = 0; i < allConfidenceValues.size(); ++i)
	{
		if (i == 0)
		{
			blocksContainTargets.push_back(cv::Point(allConfidenceValues[i].x, allConfidenceValues[i].y));
			currentTargetCountIndex++;
			continue;
		}
		if (allConfidenceValues[i].confidenceVal > 5 && allConfidenceValues[i].confidenceVal == allConfidenceValues[i - 1].confidenceVal)
		{
			blocksContainTargets.push_back(cv::Point(allConfidenceValues[i].x, allConfidenceValues[i].y));
		}
		else
		{
			if (allConfidenceValues[i].confidenceVal >= 5)
			{
				blocksContainTargets.push_back(cv::Point(allConfidenceValues[i].x, allConfidenceValues[i].y));
				currentTargetCountIndex++;
				if (currentTargetCountIndex >= maxTargetCount)
					break;
			}
		}
	}

	std::cout << "All Blocks:" << std::endl;
	for (auto point : blocksContainTargets)
		std::cout << "X = " << point.x << " Y = " << point.y << std::endl;
}

void CheckTrackerForThisBlock(cv::Point blockPos, int& trackerIndex)
{
	for (auto j = 0; j < GlobalTrackerList.size(); ++j)
	{
		if (GlobalTrackerList[j].blockX == blockPos.x && GlobalTrackerList[j].blockY == blockPos.y)
		{
			trackerIndex = j + 1;
			break;
		}
	}
}

void CreateNewTrackerForThisBlock(cv::Point blockPos, cv::Rect rect, const cv::Mat& frame)
{
	TargetTracker tracker;
	tracker.blockX = blockPos.x;
	tracker.blockY = blockPos.y;
	tracker.leftTopX = rect.x;
	tracker.leftTopY = rect.y;
	tracker.targetRect = rect;
	tracker.timeLeft = 1;

	//	tracker.feature.clear();
	//	tracker.feature = Util::ToFeatureVector(frame(rect));

	GlobalTrackerList.push_back(tracker);
}

void ConfidenceValueLost(std::vector<std::vector<int>> confidenceValueMap)
{
	for (auto x = 0; x < countX; ++x)
	{
		for (auto y = 0; y < countY; ++y)
		{
			if (confidenceValueMap[y][x] > 0)
			{
				confidenceValueMap[y][x] -= 3;
				if (confidenceValueMap[y][x] < 0)
					confidenceValueMap[y][x] = 0;
			}
		}
	}
}

bool ReSearchTarget(const cv::Mat& curFrame, TargetTracker& tracker)
{
	auto leftTopBlockX = tracker.blockX - 1 >= 0 ? tracker.blockX - 1 : tracker.blockX;
	auto leftTopBlockY = tracker.blockY - 1 >= 0 ? tracker.blockY - 1 : tracker.blockY;

	auto rightBottomBlockX = tracker.blockX + 1 < countX ? tracker.blockX + 1 : tracker.blockX;
	auto rightBottomBlockY = tracker.blockY + 1 < countY ? tracker.blockY + 1 : tracker.blockY;

	auto leftTopX = leftTopBlockX * STEP;
	auto leftTopY = leftTopBlockY * STEP;

	auto rightBottomX = (rightBottomBlockX + 1) * STEP - 1;
	if (rightBottomX >= 320)
		rightBottomX = 319;
	auto rightBottomY = (rightBottomBlockY + 1) * STEP - 1;
	if (rightBottomY >= 256)
		rightBottomY = 255;

	auto width = tracker.targetRect.width;
	auto height = tracker.targetRect.height;

	auto minDiff = width * height * height * width;
	auto minRow = -1;
	auto minCol = -1;
	std::vector<uchar> minFeature(width * height, 0);

	for (auto row = leftTopY; row + height - 1 <= rightBottomY; ++ row)
	{
		for (auto col = leftTopX; col + width - 1 <= rightBottomX; ++col)
		{
			auto feature = Util::ToFeatureVector(curFrame(cv::Rect(col, row, width, height)));

			auto curDiff = Util::FeatureDiff(feature, tracker.feature);
			if (curDiff < minDiff)
			{
				minDiff = curDiff;
				minRow = row;
				minCol = col;
				minFeature = feature;
			}
		}
	}

	if (minCol != -1)
	{
		tracker.targetRect = cv::Rect(minCol, minRow, width, height);
		tracker.leftTopX = minCol;
		tracker.leftTopY = minRow;
		tracker.blockX = minCol / STEP;
		tracker.blockY = minRow / STEP;
		tracker.feature = minFeature;
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

				auto targetRects = DetectByMaxFilterAndAdptiveThreshold::Detect(curFrame);

				UpdateConfidenceMap(queueEndIndex, confidenceQueueMap, targetRects, Four);

				UpdateConfidenceVector(countX, countY, confidenceQueueMap, allConfidenceQueue);

				auto searchIndex = 0;

				GetMostLiklyTargetsRect(allConfidenceQueue, TopCount, searchIndex);

				DrawRectangleForAllCandidateTargets(colorFrame, allConfidenceQueue, targetRects, searchIndex, confidenceValueMap);

				PrintConfidenceValueMap(confidenceValueMap, "Before Draw Rect");

				if (frameIndex > ThinkingTime)
				{
					const auto maxTargetCount = 2;
					std::vector<cv::Point> blocksContainTargets;

					UpdateConfidenceValueVector(confidenceValueMap, allConfidenceValues);

					GetTopCountBlocksWhichContainsTargets(allConfidenceValues, maxTargetCount, blocksContainTargets);

					for (auto i = 0; i < blocksContainTargets.size(); ++i)
					{
						auto findTargetFlag = false;
						auto trackerIndex = 0;

						auto currentBlock = blocksContainTargets[i];

						CheckTrackerForThisBlock(currentBlock, trackerIndex);

						for (auto j = 0; j < targetRects.size(); ++j)
						{
							auto rect = targetRects[j];
							auto x = (rect.x + rect.width / 2) / STEP;
							auto y = (rect.y + rect.height / 2) / STEP;

							if (x == currentBlock.x && y == currentBlock.y)
							{
								if (GlobalTrackerList.empty())
								{
									CreateNewTrackerForThisBlock(currentBlock, rect, curFrame);
								}
								else
								{
									if (!TrackerDecited(rect, x, y, trackerIndex, curFrame))
									{
										CreateNewTrackerForThisBlock(currentBlock, rect, curFrame);
									}
								}

								findTargetFlag = true;
							}
							else if ((x - 1 >= 0 && x - 1 == currentBlock.x && y == currentBlock.y) ||
								(y - 1 >= 0 && x == currentBlock.x && y - 1 == currentBlock.y) ||
								(x + 1 < countX && x + 1 == currentBlock.x && y == currentBlock.y) ||
								(y + 1 < countY && x == currentBlock.x && y + 1 == currentBlock.y) ||
								(y + 1 < countY && x + 1 < countX && x + 1 == currentBlock.x && y + 1 == currentBlock.y) ||
								(y + 1 < countY && x - 1 >= 0 && x - 1 == currentBlock.x && y + 1 == currentBlock.y) ||
								(y - 1 < countY && x + 1 < countX && x + 1 == currentBlock.x && y - 1 == currentBlock.y) ||
								(y - 1 < countY && x - 1 >= 0 && x - 1 == currentBlock.x && y - 1 == currentBlock.y))
							{
								confidenceValueMap[currentBlock.y][currentBlock.x] = MinNeighbor(confidenceValueMap, currentBlock.y, currentBlock.x);
								confidenceValueMap[y][x] = MaxNeighbor(confidenceValueMap, currentBlock.y, currentBlock.x);

								if (GlobalTrackerList.empty())
								{
									CreateNewTrackerForThisBlock(cv::Point(x, y), rect, curFrame);
								}
								else
								{
									if (!TrackerDecited(rect, x, y, trackerIndex, curFrame))
									{
										CheckTrackerForThisBlock(cv::Point(x, y), trackerIndex);
										if (!TrackerDecited(rect, x, y, trackerIndex, curFrame))
											CreateNewTrackerForThisBlock(cv::Point(x, y), rect, curFrame);
									}

									findTargetFlag = true;
								}
							}
						}

						if (!findTargetFlag)
						{
							if (trackerIndex > 0)
							{
								auto tracker = GlobalTrackerList[trackerIndex - 1];

								tracker.timeLeft--;
								if (tracker.timeLeft == 0)
								{
									auto it = GlobalTrackerList.begin() + (trackerIndex - 1);

									auto col = it->blockX;
									auto row = it->blockY;

									confidenceValueMap[row][col] /= 2;
									if (col - 1 >= 0)
										confidenceValueMap[row][col - 1] /= 2;
									if (col + 1 < countX)
										confidenceValueMap[row][col + 1] /= 2;
									if (row - 1 >= 0)
										confidenceValueMap[row - 1][col] /= 2;
									if (row + 1 < countY)
										confidenceValueMap[row + 1][col] /= 2;

									GlobalTrackerList.erase(it);
								}
							}
							else
							{
								auto col = currentBlock.x;
								auto row = currentBlock.y;

								confidenceValueMap[row][col] /= 2;
								if (col - 1 >= 0)
									confidenceValueMap[row][col - 1] /= 2;
								if (col + 1 < countX)
									confidenceValueMap[row][col + 1] /= 2;
								if (row - 1 >= 0)
									confidenceValueMap[row - 1][col] /= 2;
								if (row + 1 < countY)
									confidenceValueMap[row + 1][col] /= 2;
							}
						}
					}

					std::cout << "All Tracker" << std::endl;
					for (auto tracker : GlobalTrackerList)
					{
						std::cout << "X = " << tracker.blockX << " Y = " << tracker.blockY << " Time Left = " << tracker.timeLeft << std::endl;
					}

					for (auto it = GlobalTrackerList.begin(); it != GlobalTrackerList.end(); ++it)
					{
						std::cout << "Test Tracker" << std::endl;
						std::cout << "X = " << it->blockX << " Y = " << it->blockY << std::endl;

						auto existFlag = false;
						for (auto target : blocksContainTargets)
						{
							std::cout << "Current X = " << target.x << " Current Y = " << target.y << std::endl;
							if (it->blockX == target.x && it->blockY == target.y)
								existFlag = true;
						}

						if (!existFlag)
						{
							it->timeLeft--;
							if (it->timeLeft == 0)
							{
								auto col = it->blockX;
								auto row = it->blockY;

								confidenceValueMap[row][col] /= 2;
								if (col - 1 >= 0)
									confidenceValueMap[row][col - 1] /= 2;
								if (col + 1 < countX)
									confidenceValueMap[row][col + 1] /= 2;
								if (row - 1 >= 0)
									confidenceValueMap[row - 1][col] /= 2;
								if (row + 1 < countY)
									confidenceValueMap[row + 1][col] /= 2;

								it = GlobalTrackerList.erase(it);
								if (it == GlobalTrackerList.end())
									break;
							}
						}
					}

					//					PrintConfidenceValueMap(confidenceValueMap, "After Draw Rect");

					ConfidenceValueLost(confidenceValueMap);

					sort(GlobalTrackerList.begin(), GlobalTrackerList.end(), Util::CompareTracker);
					for (auto tracker : GlobalTrackerList)
					{
						if (tracker.timeLeft > 2)
							rectangle(colorFrame, cv::Rect(tracker.targetRect.x - 2, tracker.targetRect.y - 2, tracker.targetRect.width + 4, tracker.targetRect.height + 4), tracker.Color());
					}
					ConfidenceMapUtil::LostMemory(countX, countY, queueSize, queueEndIndex, confidenceQueueMap);
				}
				imshow("last result", colorFrame);
				if (frameIndex == 0)
					cv::waitKey(0);

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
