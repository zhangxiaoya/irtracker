#pragma once
#include "../Headers/GlobalConstantConfigure.h"
#include "../FrameSource/FrameSource.hpp"

static std::string inFullStr;
static std::string outFullStr;

static bool ImageListReadFlag = false;

const char* GlobalImageListNameFormat;
const char* GlobalWriteFileNameFormat;

inline void ForTwoBins()
{
	std::string listNum = "2";

	std::string inPrefix = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_";
	std::string inBackend = "_8bit\\Frame_%04d.png";
	inFullStr = inPrefix + listNum + inBackend;

	GlobalImageListNameFormat = inFullStr.c_str();

	std::string outPrefix = ".\\ir_file_20170531_1000m_";
	std::string outBackend = "\\Frame_%04d.png";
	outFullStr = outPrefix + listNum + outBackend;

	GlobalWriteFileNameFormat = outFullStr.c_str();
}

inline cv::Mat GetTheFirstImage(int firImageIndex = 0)
{
	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];
	sprintf_s(imageFullName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalImageListNameFormat, firImageIndex);

	return cv::imread(imageFullName);
}

inline void UpdateImageSize(cv::Mat& img)
{
	std::cout << "Update Image Size" << std::endl;

	IMAGE_WIDTH = img.cols;
	IMAGE_HEIGHT = img.rows;
	countX = ceil(static_cast<double>(IMAGE_WIDTH) / BLOCK_SIZE);
	countY = ceil(static_cast<double>(IMAGE_HEIGHT) / BLOCK_SIZE);
}

inline void UpdateDataType(const cv::Mat& img)
{
	std::cout << "Update Image Size" << std::endl;

	CV_DATA_TYPE = img.type();
}

inline bool UpdateConstants()
{
	auto img = GetTheFirstImage();
	if (!img.empty())
	{
		UpdateImageSize(img);

		UpdateDataType(img);

		return true;
	}

	std::cout << "Read Image Failed, please check whether the path exist!" << std::endl;
	return false;
}

inline void InitConfigure()
{
//	ForTwoBins();

	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1_0-600m_150ms\\Frame_%08d.bmp";
	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1_0-600m_150ms\\Frame_%08d.bmp";

	// GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1km\\images\\Frame_%08d.png";
	// GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1km\\Frame_%08d.png";

	// GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\2_500-1500_150ms\\Frame_%08d.bmp";
	// GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\2_500-1500_150ms\\Frame_%08d.png";

	// GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\3_1500m_100ms\\images\\Frame_%08d.png";
	// GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\3_1500m_100ms\\Frame_%08d.png";

	// GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1500_middle\\images\\Frame_%08d.png";
	// GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500_middle\\Frame_%08d.png";

	// GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1500-700_middle\\images\\Frame_%08d.png";
	// GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500-700_middle\\Frame_%08d.png";

	if(UpdateConstants())
	{
		ImageListReadFlag = true;
	}
	else
	{
		ImageListReadFlag = false;
	}
}
