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
	
//		firstImageList = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_1_8bit_maxFilter_discrezated\\Frame_%04d.png";
		firstImageList = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";
//		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit_maxFilter\\Frame_%04d.png";
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
	cv::Mat colorFrame;

	auto frameIndex = 0;

	char *fileNameFormat = ".\\ir_file_20170531_1000m_1\\Frame_%04d.png";
	const auto bufferSize = 100;
	char fileName[bufferSize];

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

				cvtColor(curFrame, colorFrame, CV_GRAY2BGR);

//				DetectByDiscontinuity::Detect(curFrame);

//				DetectByBinaryBitMap::Detect(curFrame);

//				DetectByMultiScaleLocalDifference::Detect(curFrame);

				auto targetRects = DetectByMaxFilterAndAdptiveThreshHold::Detect(curFrame);

				for (auto i = 0; i < targetRects.size(); ++i)
					rectangle(colorFrame, cv::Rect(targetRects[i].x-1,targetRects[i].y-1,targetRects[i].width+2,targetRects[i].height+2), cv::Scalar(255, 255, 0));

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
