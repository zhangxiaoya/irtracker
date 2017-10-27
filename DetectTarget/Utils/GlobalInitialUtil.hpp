#pragma once
#include "../Headers/GlobalConstantConfigure.h"

static std::string inFullStr;
static std::string outFullStr;
static std::string outFolderStr;

static bool ImageListReadFlag = false;

const char* GlobalImageListNameFormat;
const char* GlobalWriteFileNameFormat;
const char* GlobalWriteVideoFileFolder;

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
	if(colorImg.empty() == true)
	{
		logPrinter.PrintLogs("Get the first image frame failed, cannot update parameters!", LogLevel::Error);
		return false;
	}
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

enum SecondTestCaseListNum
{
	First,
	Second
};

inline void ForSecondOriginalBinFiles(SecondTestCaseListNum listNum, bool isShowLastResultOnly = false)
{
	SHOW_LAST_RESULT_ONLY = isShowLastResultOnly;
	std::string listNumStr;

	switch (listNum)
	{
	case First:
		{
			listNumStr = "1";
			break;
		}
	case Second:
		{
			listNumStr = "2";
			break;
		}
	default:
		{
			logPrinter.PrintLogs("Image List Index Out Of Range, Please Check It!", LogLevel::Error);
			return;
			break;
		}
	}

	CHECK_SURROUNDING_BOUNDARY_FLAG = false;
	CHECK_INSIDE_BOUNDARY_FLAG = false;
	CHECK_FOUR_BLOCK_FLAG = false;
	CHECK_COVERAGE_FLAG = false;

	CHECK_ORIGIN_FLAG = true;
	ConvexPartitionOfOriginalImage = 4;
	ConcavePartitionOfOriginalImage = 1;

	CHECK_DECRETIZATED_FLAG = true;
	ConvexPartitionOfDiscretizedImage = 4;
	ConcavePartitionOfDiscretizedImage = 1;
	IsNeedStrengthenIntensity = false;

	CHECK_STANDARD_DEVIATION_FLAG = true;

	std::string inPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Second\\ir_file_20170531_1000m_";
	std::string inBackend = "_8bit\\Frame_%04d.png";
//	std::string inPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Second\\Frames\\ir_file_20170531_1000m_";
//	std::string inBackend = "\\Frame_%08d.png";
	inFullStr = inPrefix + listNumStr + inBackend;

	GlobalImageListNameFormat = inFullStr.c_str();

//	std::string outPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Second\\newResults\\ir_file_20170531_1000m_";
//	std::string outBackend = "\\Frame_%04d.png";
	std::string outPrefix = "E:\\WorkLogs\\Projects\\Project4\\Data\\Second\\FramesResults\\ir_file_20170531_1000m_";
	std::string outBackend = "\\Frame_%08d.png";
	outFullStr = outPrefix + listNumStr + outBackend;

	GlobalWriteFileNameFormat = outFullStr.c_str();

	outFolderStr = outPrefix + listNumStr;
	GlobalWriteVideoFileFolder = outFolderStr.c_str();
}

inline void ForWorstFrames(std::string distance, std::string flyStatus, bool isShowLastResultOnly = false)
{
	SHOW_LAST_RESULT_ONLY = isShowLastResultOnly;
	CHECK_SURROUNDING_BOUNDARY_FLAG = true;
	CHECK_INSIDE_BOUNDARY_FLAG = true;
	CHECK_FOUR_BLOCK_FLAG = false;
	CHECK_COVERAGE_FLAG = false;

	CHECK_ORIGIN_FLAG = true;
	ConvexPartitionOfOriginalImage = 10;
	ConcavePartitionOfOriginalImage = 1;

	CHECK_DECRETIZATED_FLAG = true;
	ConvexPartitionOfDiscretizedImage = 10;
	ConcavePartitionOfDiscretizedImage = 1;

	IsNeedStrengthenIntensity = false;
	CHECK_STANDARD_DEVIATION_FLAG = true;

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

enum TargetMoveDirection
{
	In,
	Out
};

inline void For6kmData(TargetMoveDirection direction, bool isShowLastResultOnly = false)
{
	SHOW_LAST_RESULT_ONLY = isShowLastResultOnly;
	CHECK_STANDARD_DEVIATION_FLAG = false;
	CHECK_SURROUNDING_BOUNDARY_FLAG = true;
	CHECK_INSIDE_BOUNDARY_FLAG = true;
	CHECK_FOUR_BLOCK_FLAG = false;
	CHECK_COVERAGE_FLAG = false;

	CHECK_ORIGIN_FLAG = true;
	ConvexPartitionOfOriginalImage = 10;
	ConcavePartitionOfOriginalImage = 1;

	CHECK_DECRETIZATED_FLAG = false;
	ConvexPartitionOfDiscretizedImage = 5;
	ConcavePartitionOfDiscretizedImage = 1;

	IsNeedStrengthenIntensity = false;

	if (direction == In)
	{
		GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\in\\target_in_6km_%02d.png";
		GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\results\\in\\target_in_6km_1%02d.png";
		GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Fifth\\6km\\Frames\\results\\in";
	}
	else if (direction == Out)
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

enum LongWaveEnum
{
	fragment_1_0_600m_150ms,
	fragment_1km,
	fragment_2_500_1500_150ms,
	fragment_3_1500m_100ms,
	fragment_1500_middle,
	fragment_1500_700_middle
};

inline void ForbackgroundData()
{
	CHECK_STANDARD_DEVIATION_FLAG = true;
	CHECK_SURROUNDING_BOUNDARY_FLAG = true;
	CHECK_INSIDE_BOUNDARY_FLAG = true;
	CHECK_FOUR_BLOCK_FLAG = false;
	CHECK_COVERAGE_FLAG = true;

	CHECK_ORIGIN_FLAG = true;
	ConvexPartitionOfOriginalImage = 10;
	ConcavePartitionOfOriginalImage = 1;

	CHECK_DECRETIZATED_FLAG = true;
	ConvexPartitionOfDiscretizedImage = 10;
	ConcavePartitionOfDiscretizedImage = 1;

	IsNeedStrengthenIntensity = false;
	DISCRATED_BIN = 15;
	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Sixth\\backGround\\images\\Frame_%08d.png";
	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Sixth\\result\\Frame_%08d.png";
	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Sixth\\result";
}

inline void ForThirdLongWave(LongWaveEnum fragmentKind, bool isShowLastResultOnly = false)
{
	SHOW_LAST_RESULT_ONLY = isShowLastResultOnly;
	CHECK_STANDARD_DEVIATION_FLAG = false;
	switch (fragmentKind)
	{
	case fragment_1_0_600m_150ms:
		{
			CHECK_SURROUNDING_BOUNDARY_FLAG = false;
			CHECK_INSIDE_BOUNDARY_FLAG = false;
			CHECK_FOUR_BLOCK_FLAG = false;
			CHECK_COVERAGE_FLAG = false;

			CHECK_ORIGIN_FLAG = true;
			ConvexPartitionOfOriginalImage = 15;
			ConcavePartitionOfOriginalImage = 1;

			CHECK_DECRETIZATED_FLAG = true;
			ConvexPartitionOfDiscretizedImage = 5;
			ConcavePartitionOfDiscretizedImage = 1;

			IsNeedStrengthenIntensity = true;
			DISCRATED_BIN = 10;
			GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1_0-600m_150ms\\Frame_%08d.bmp";
			GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1_0-600m_150ms\\Frame_%08d.bmp";
			GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1_0-600m_150ms";
			break;
		}
	case fragment_1km:
		{
			CHECK_SURROUNDING_BOUNDARY_FLAG = true;
			CHECK_INSIDE_BOUNDARY_FLAG = true;
			CHECK_FOUR_BLOCK_FLAG = false;
			CHECK_COVERAGE_FLAG = true;

			CHECK_ORIGIN_FLAG = true;
			ConvexPartitionOfOriginalImage = 10;
			ConcavePartitionOfOriginalImage = 1;

			CHECK_DECRETIZATED_FLAG = true;
			ConvexPartitionOfDiscretizedImage = 5;
			ConcavePartitionOfDiscretizedImage = 1;

			IsNeedStrengthenIntensity = false;
			GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1km\\images\\Frame_%08d.png";
			GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1km\\Frame_%08d.png";
			GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1km";
			break;
		}
	case fragment_2_500_1500_150ms:
		{
			CHECK_SURROUNDING_BOUNDARY_FLAG = false;
			CHECK_INSIDE_BOUNDARY_FLAG = true;
			CHECK_FOUR_BLOCK_FLAG = false;
			CHECK_COVERAGE_FLAG = false;

			CHECK_ORIGIN_FLAG = true;
			ConvexPartitionOfOriginalImage = 10;
			ConcavePartitionOfOriginalImage = 1;

			CHECK_DECRETIZATED_FLAG = true;
			ConvexPartitionOfDiscretizedImage = 20;
			ConcavePartitionOfDiscretizedImage = 1;

			IsNeedStrengthenIntensity = true;

			GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\2_500-1500_150ms\\Frame_%08d.bmp";
			GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\2_500-1500_150ms\\Frame_%08d.png";
			GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\2_500-1500_150ms";
			break;
		}
	case fragment_3_1500m_100ms:
		{
			CHECK_SURROUNDING_BOUNDARY_FLAG = true;
			CHECK_INSIDE_BOUNDARY_FLAG = true;
			CHECK_FOUR_BLOCK_FLAG = false;
			CHECK_COVERAGE_FLAG = false;

			CHECK_ORIGIN_FLAG = true;
			ConvexPartitionOfOriginalImage = 6;
			ConcavePartitionOfOriginalImage = 1;

			CHECK_DECRETIZATED_FLAG = true;
			ConvexPartitionOfDiscretizedImage = 2;
			ConcavePartitionOfDiscretizedImage = 1;

			IsNeedStrengthenIntensity = false;
			DISCRATED_BIN = 20;

			GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\3_1500m_100ms\\images\\Frame_%08d.png";
			GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\3_1500m_100ms\\Frame_%08d.png";
			GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\3_1500m_100ms";
			break;
		}
	case fragment_1500_middle:
		{
			CHECK_SURROUNDING_BOUNDARY_FLAG = false;
			CHECK_INSIDE_BOUNDARY_FLAG = true;
			CHECK_FOUR_BLOCK_FLAG = false;
			CHECK_COVERAGE_FLAG = false;

			CHECK_ORIGIN_FLAG = true;
			ConvexPartitionOfOriginalImage = 15;
			ConcavePartitionOfOriginalImage = 1;

			CHECK_DECRETIZATED_FLAG = true;
			ConvexPartitionOfDiscretizedImage = 5;
			ConcavePartitionOfDiscretizedImage = 1;

			IsNeedStrengthenIntensity = false;
			DISCRATED_BIN = 20;

			GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1500_middle\\images\\Frame_%08d.png";
			GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500_middle\\Frame_%08d.png";
			GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500_middle";
			break;
		}
	case fragment_1500_700_middle:
		{
			CHECK_SURROUNDING_BOUNDARY_FLAG = true;
			CHECK_INSIDE_BOUNDARY_FLAG = true;
			CHECK_FOUR_BLOCK_FLAG = false;
			CHECK_COVERAGE_FLAG = true;

			CHECK_ORIGIN_FLAG = true;
			ConvexPartitionOfOriginalImage = 10;
			ConcavePartitionOfOriginalImage = 1;

			CHECK_DECRETIZATED_FLAG = true;
			ConvexPartitionOfDiscretizedImage = 5;
			ConcavePartitionOfDiscretizedImage = 1;

			IsNeedStrengthenIntensity = false;
			GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\1500-700_middle\\images\\Frame_%08d.png";
			GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500-700_middle\\Frame_%08d.png";
			GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\Third\\Raw\\result\\1500-700_middle";
			break;
		}
	default:
		{
			auto logMsg = "Not Found Fragment";
			logPrinter.PrintLogs(logMsg, LogLevel::Error);
			break;
		}
	}
}

inline void ForDDEBackGroundData()
{
	CHECK_STANDARD_DEVIATION_FLAG = false;
	CHECK_SURROUNDING_BOUNDARY_FLAG = false;
	CHECK_INSIDE_BOUNDARY_FLAG = false;
	CHECK_FOUR_BLOCK_FLAG = false;
	CHECK_COVERAGE_FLAG = false;

	CHECK_CONSTRAST = true;

	CHECK_ORIGIN_FLAG = true;
	ConvexPartitionOfOriginalImage = 200;
	ConcavePartitionOfOriginalImage = 1;

	CHECK_DECRETIZATED_FLAG = true;
	ConvexPartitionOfDiscretizedImage = 200;
	ConcavePartitionOfDiscretizedImage = 1;

	IsNeedStrengthenIntensity = false;
	DISCRATED_BIN = 10;
	GlobalImageListNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\video\\frames\\Frame_%08d.png";
	GlobalWriteFileNameFormat = "E:\\WorkLogs\\Projects\\Project4\\Data\\video\\result\\Frame_%08d.png";
	GlobalWriteVideoFileFolder = "E:\\WorkLogs\\Projects\\Project4\\Data\\video\\result";
}

inline void InitGlobalConfigure()
{
	//ForSecondOriginalBinFiles(Second, true);

//	ForWorstFrames("500", "yundong");

	//ForThirdLongWave(fragment_1km, true);

	//For6kmData(Out, true);

//	ForbackgroundData();

	ForDDEBackGroundData();

	if (UpdateConstants())
	{
		ImageListReadFlag = true;
	}
	else
	{
		ImageListReadFlag = false;
	}
}
