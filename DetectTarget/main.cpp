#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <iterator>

#include "DetectByMaxFilterAndAdptiveThreshHold.hpp"

const auto SHOW_DELAY = 1;

void InitVideoReader(cv::VideoCapture& video_capture)
{
	const char* firstImageList;

	if(AFTER_MAX_FILTER)
	{
		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit_maxFilter\\Frame_%04d.png";
//		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter\\Frame_%04d.png";
	}
	else
	{
		// firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";
		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit\\Frame_%04d.png";
	}
	video_capture.open(firstImageList);
}

void RemoveUnusedPixel(cv::Mat curFrame)
{
	for (auto r = 0; r < 2; ++r)
		for(auto c = 0;c<11;++c)
			curFrame.at<uchar>(r, c) = 0;
}

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture;
	InitVideoReader(video_capture);

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
				RemoveUnusedPixel(curFrame);

				imshow("Current Frame", curFrame);
				cv::waitKey(SHOW_DELAY);

//				DetectByDiscontinuity::Detect(curFrame);

//				DetectByBinaryBitMap::Detect(curFrame);

//				DetectByMultiScaleLocalDifference::Detect(curFrame);

				DetectByMaxFilterAndAdptiveThreshHold::Detect(curFrame);

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
