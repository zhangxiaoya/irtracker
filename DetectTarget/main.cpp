#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include  <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <stack>

auto const REDCOLOR = cv::Scalar(0, 0, 255);
auto const BLUECOLOR = cv::Scalar(255, 0, 0);
auto const GREENCOLOR = cv::Scalar(0, 255, 0);

const auto DELAY = 100;
const auto WINDOW_WIDTH = 8;
const auto WINDOW_HEIGHT = 8;

const auto THRESHHOLD = 20;
const auto CONTIUNITY_THRESHHOLD = 0.4;

const auto TARGET_WIDTH_MIN_LIMIT = 4;
const auto TARGET_HEIGHT_MIN_LIMIT = 4;
const auto TARGET_WIDTH_MAX_LIMIT = 16;
const auto TARGET_HEIGHT_MAX_LIMIT = 16;

struct FourLimits
{
	FourLimits():top(-1),bottom(-1),left(-1),right(-1){}

	int top;
	int bottom;
	int left;
	int right;
};

enum FieldType
{
	Four,
	Eight
};

const char* firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\second\\frame_%04d.png";

void BinaryMat(cv::Mat& mat)
{
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			mat.at<uchar>(r, c) = mat.at<uchar>(r, c) > THRESHHOLD ? 1 : 0;
}

double MeanMat(const cv::Mat& mat)
{
	double sum = 0;
	for (auto r = 0; r < mat.rows; ++r)
		for (auto c = 0; c < mat.cols; ++c)
			sum += static_cast<int>(mat.at<uchar>(r, c));

	return  sum / (mat.rows * mat.cols);
}

bool CheckDiscontinuity(const cv::Mat& frame, const cv::Point& leftTop)
{
	auto curRect = cv::Rect(leftTop.x, leftTop.y, WINDOW_WIDTH, WINDOW_HEIGHT);
	cv::Mat curMat;
	frame(curRect).copyTo(curMat);

	auto regionMean = MeanMat(curMat);

	BinaryMat(curMat);

	auto rowTop = leftTop.y - 1;
	auto rowBottom = leftTop.y + WINDOW_HEIGHT;
	auto colLeft = leftTop.x - 1;
	auto colRight = leftTop.x + WINDOW_WIDTH;

	auto totalCount = 0;
	auto continuityCount = 0;

	auto sum = 0.0;

	if(rowTop >= 0)
	{
		for (auto x = leftTop.x; x < leftTop.x + WINDOW_WIDTH; ++x)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(rowTop, x));

			auto curValue = frame.at<uchar>(rowTop, x) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(rowTop - leftTop.y + 1, x - leftTop.x))
				continuityCount++;
		}
	}
	if (rowBottom < frame.rows)
	{
		for (auto x = leftTop.x; x < leftTop.x + WINDOW_WIDTH; ++x)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(rowBottom, x));

			auto curValue = frame.at<uchar>(rowBottom, x) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(rowBottom - leftTop.y - 1, x - leftTop.x))
				continuityCount++;
		}
	}

	if(colLeft >=0)
	{
		for (auto y = leftTop.y; y < leftTop.y + WINDOW_HEIGHT; ++y)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(y, colLeft));

			auto curValue = frame.at<uchar>(y, colLeft) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(y - leftTop.y, colLeft - leftTop.x + 1))
				continuityCount++;
		}
	}

	if(colRight < frame.cols)
	{
		for (auto y = leftTop.y; y < leftTop.y + WINDOW_HEIGHT; ++y)
		{
			totalCount++;
			sum += static_cast<int>(frame.at<uchar>(y, colRight));

			auto curValue = frame.at<uchar>(y, colRight) > THRESHHOLD ? 1 : 0;
			if (curValue == curMat.at<uchar>(y - leftTop.y, colRight - leftTop.x - 1))
				continuityCount++;
		}
	}

	auto roundMean = sum / totalCount;

	return std::abs(roundMean - regionMean) > 2 && regionMean > THRESHHOLD;

//	return static_cast<double>(continuityCount) / totalCount < CONTIUNITY_THRESHHOLD;
}

void ShowCandidateRects(const cv::Mat& grayFrame, const std::vector<cv::Rect_<int>>& candidate_rects)
{
	cv::Mat colorFrame;
	cvtColor(grayFrame, colorFrame, CV_GRAY2RGB);

	for(auto i = 0;i<candidate_rects.size();++i)
		rectangle(colorFrame, candidate_rects[i], REDCOLOR);

	imshow("Color Frame", colorFrame);
}

void DetectTarget(cv::Mat& frame)
{
	std::vector<cv::Rect> candidateRects;

	for (auto r = 0; r < frame.rows - WINDOW_HEIGHT + 1; ++r)
	{
		for (auto c = 0; c < frame.cols - WINDOW_WIDTH + 1; ++c)
		{
			if (CheckDiscontinuity(frame, cv::Point(c, r)))
				candidateRects.push_back(cv::Rect(c, r, WINDOW_WIDTH, WINDOW_HEIGHT));
		}
	}

	ShowCandidateRects(frame, candidateRects);
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
	}
}

void DFSWithoutRecursionForField(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex)
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

void FindNeighbor(const cv::Mat& binaryFrame, cv::Mat& bitMap, int r, int c, int currentIndex, FieldType fieldType)
{
	if(fieldType == FieldType::Eight)
		DFSWithoutRecursionEightField(binaryFrame, bitMap, r, c, currentIndex);
	else if(fieldType == FieldType::Four)
		DFSWithoutRecursionForField(binaryFrame, bitMap, r, c, currentIndex);
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

void ShowCandidateTargets(const cv::Mat& curFrame, const std::vector<FourLimits>& allObject)
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

		if((width < TARGET_WIDTH_MIN_LIMIT && height < TARGET_HEIGHT_MIN_LIMIT) ||
		   (width > TARGET_WIDTH_MAX_LIMIT && height > TARGET_HEIGHT_MAX_LIMIT))
			continue;

		auto rect = cv::Rect(allObject[i].left, allObject[i].top, width, height);
		rectangle(colorFrame, rect, GREENCOLOR);
	}

	imshow("Candidate Targets", colorFrame);
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

//				DetectTarget(curFrame);

				cv::Mat binaryFrame;
				curFrame.copyTo(binaryFrame);
				BinaryMat(binaryFrame);

				cv::Mat bitMap(cv::Size(binaryFrame.cols, binaryFrame.rows), CV_32SC1, cv::Scalar(-1));
				auto totalObject = GetBitMap(binaryFrame, bitMap);

				std::vector<FourLimits> allObjects(totalObject);
				GetRectangleSize(bitMap,allObjects,totalObject);

				ShowAllObject(curFrame,allObjects);
				ShowCandidateTargets(curFrame, allObjects);

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
