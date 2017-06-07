#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>

const auto DELAY = 300;

const char* firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\second\\frame_%04d.png";

int main(int argc, char* argv[])
{
	cv::VideoCapture video_capture(firstImageList);

	cv::Mat curFrame;
	auto index = 0;

	if(video_capture.isOpened())
	{
		std::cout << "Open Image List Success!" << std::endl;

		while (!curFrame.empty() || index == 0)
		{
			video_capture >> curFrame;
			if(!curFrame.empty())
			{
				imshow("Current Frame", curFrame);
				cv::waitKey(DELAY);
				std::cout << "Index : " << std::setw(4) << index << std::endl;
				++index;
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
