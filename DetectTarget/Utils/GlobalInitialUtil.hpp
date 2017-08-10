#pragma once
#include "../Headers/GlobalConstantConfigure.h"

static std::string inFullStr;
static std::string outFullStr;
static std::string outFolderStr;

static bool ImageListReadFlag = false;

const char* GlobalImageListNameFormat;
const char* GlobalWriteFileNameFormat;
const char* GlobalWriteVideoFileFolder;

inline void ForSecondOriginalBinFiles()
{
	std::string listNum = "2";

	std::string inPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Second\\ir_file_20170531_1000m_";
//	std::string inPrefix = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_";
	std::string inBackend = "_8bit\\Frame_%04d.png";
	inFullStr = inPrefix + listNum + inBackend;

	GlobalImageListNameFormat = inFullStr.c_str();

	std::string outPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Second\\newResults\\ir_file_20170531_1000m_";
//	std::string outPrefix = "D:\\Bag\\Code_VS15\\Data\\result_";
	std::string outBackend = "\\Frame_%04d.png";
	outFullStr = outPrefix + listNum + outBackend;

	GlobalWriteFileNameFormat = outFullStr.c_str();

	outFolderStr = outPrefix + listNum;
	GlobalWriteVideoFileFolder = outFolderStr.c_str();
}

inline void ForWorstFrames(std::string distance, std::string flyStatus)
{
	std::string inPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Forth\\test\\Frames\\ir_file_20170713_";
	std::string medianPart = "m_";
	std::string inBackend = "\\Frame_%08d.png";
	inFullStr = inPrefix + distance + medianPart + flyStatus + inBackend;

	GlobalImageListNameFormat = inFullStr.c_str();

	std::string outPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Forth\\test\\Results\\ir_file_20170713_";
	std::string outBackend = "\\Frame_%08d.png";
	outFullStr = outPrefix + distance + medianPart + flyStatus + outBackend;

	GlobalWriteFileNameFormat = outFullStr.c_str();

	outFolderStr = outPrefix + distance + medianPart + flyStatus;
	GlobalWriteVideoFileFolder = outFolderStr.c_str();
}

inline cv::Mat GetTheFirstImage(int firImageIndex = 0)
{
	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];
	sprintf_s(imageFullName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalImageListNameFormat, firImageIndex);

	return cv::imread(imageFullName);
}

inline void UpdateImageSize(cv::Mat& img)
{
	logPrinter.PrintLogs("Update Image Size", LogLevel::Info);

	IMAGE_WIDTH = img.cols;
	IMAGE_HEIGHT = img.rows;
	countX = ceil(static_cast<double>(IMAGE_WIDTH) / BLOCK_SIZE);
	countY = ceil(static_cast<double>(IMAGE_HEIGHT) / BLOCK_SIZE);
}

inline void UpdateDataType(const cv::Mat& img)
{
	logPrinter.PrintLogs("Update Image DataType", LogLevel::Info);

	CV_DATA_TYPE = img.type();
}

inline bool UpdateConstants()
{
	auto colorImg = GetTheFirstImage();
	cv::Mat grayImg;
	cvtColor(colorImg, grayImg, CV_RGB2GRAY);

	if (!colorImg.empty())
	{
		UpdateImageSize(grayImg);

		UpdateDataType(grayImg);

		return true;
	}

	logPrinter.PrintLogs("Read Image Failed, please check whether the path exist!", LogLevel::Error);
	return false;
}

enum TargetMoveDirection
{
	In,
	Out
};

inline void For6kmData(TargetMoveDirection direction)
{
	if(direction == In)
	{
		GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\in\\target_in_6km_%02d.png";
		GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\results\\in\\target_in_6km_1%02d.png";
		GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\results\\in";
	}
	else if(direction == Out)
	{
		GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\out\\target_out_6km_%02d.png";
		GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\results\\out\\target_out_6km_1%02d.png";
		GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\results\\out";
	}
	else
	{
		logPrinter.PrintLogs("Target Move Direction Operator Not Matched!", LogLevel::Error);
	}
}

inline void InitGlobalConfigure()
{
	ForSecondOriginalBinFiles();

//	ForWorstFrames("500", "jingzhi");

//	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1_0-600m_150ms\\Frame_%08d.bmp";
//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1_0-600m_150ms\\Frame_%08d.bmp";
//	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1_0-600m_150ms";

//	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1km\\images\\Frame_%08d.png";
//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1km\\Frame_%08d.png";
//	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1km";

//	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\2_500-1500_150ms\\Frame_%08d.bmp";
//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\2_500-1500_150ms\\Frame_%08d.png";
//	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\2_500-1500_150ms";

//	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\3_1500m_100ms\\images\\Frame_%08d.png";
//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\3_1500m_100ms\\Frame_%08d.png";
//	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\3_1500m_100ms";

//	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1500_middle\\images\\Frame_%08d.png";
//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500_middle\\Frame_%08d.png";
//	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500_middle";

//	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1500-700_middle\\images\\Frame_%08d.png";
//	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500-700_middle\\Frame_%08d.png";
//	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500-700_middle";

	For6kmData(In);

	if(UpdateConstants())
	{
		ImageListReadFlag = true;
	}
	else
	{
		ImageListReadFlag = false;
	}
}
