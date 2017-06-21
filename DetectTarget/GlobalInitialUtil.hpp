#pragma once
#include "ConfidenceMapUtil.hpp"
#include "GlobalConstantConfigure.h"

const char* GlobalWriteFileNameFormat;
const char* GlobalImageListFolder;
std::string inFullStr;
std::string outFullStr;

inline void UpdateImageSize()
{
	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];
	sprintf_s(imageFullName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalImageListFolder, 0);
	auto img = cv::imread(imageFullName);
	if(!img.empty())
	{
		std::cout << "Update Image Size" <<std::endl;

		IMAGE_WIDTH = img.cols;
		IMAGE_HEIGHT = img.rows;
		countX = ceil(static_cast<double>(IMAGE_WIDTH) / BLOCK_SIZE);
		countY = ceil(static_cast<double>(IMAGE_WIDTH) / BLOCK_SIZE);
	}
	else
	{
		std::cout << "Read Image Failed, please check whether the path exist!" << std::endl;
	}
}

inline void ForTwoBins()
{
	std::string listNum = "2";
	// GlobalImageListFolder = "D:\\Bag\\Code_VS15\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";

	std::string inPrefix = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_";
	std::string inBackend = "_8bit\\Frame_%04d.png";
	inFullStr = inPrefix + listNum + inBackend;

	GlobalImageListFolder = inFullStr.c_str();

	UpdateImageSize();

	std::string outPrefix = ".\\ir_file_20170531_1000m_";
	std::string outBackend = "\\Frame_%04d.png";
	outFullStr = outPrefix + listNum + outBackend;

	GlobalWriteFileNameFormat = outFullStr.c_str();
}

inline void InitVideoReader(cv::VideoCapture& video_capture)
{
	ForTwoBins();

//	GlobalImageListFolder = "E:\\WorkLogs\\Data\\Ir\\207\\Raw\\1_0-600m_150ms\\00000000_00000000001582C6_%08d.bmp";

//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Data\\Ir\\207\\Raw\\result\\1\\00000000_00000000001582C6_%08d.bmp";

	video_capture.open(GlobalImageListFolder);
}
