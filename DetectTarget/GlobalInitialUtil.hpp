#pragma once

auto const REDCOLOR = cv::Scalar(0, 0, 255);
auto const BLUECOLOR = cv::Scalar(255, 0, 0);
auto const GREENCOLOR = cv::Scalar(0, 255, 0);
auto const YELLOWCOLOR = cv::Scalar(0, 255, 255);

const auto WINDOW_WIDTH = 8;
const auto WINDOW_HEIGHT = 8;
const auto THRESHHOLD = 25;

const auto ThinkingTime = 6;

const auto TARGET_WIDTH_MIN_LIMIT = 2;
const auto TARGET_HEIGHT_MIN_LIMIT = 2;
const auto TARGET_WIDTH_MAX_LIMIT = 16;
const auto TARGET_HEIGHT_MAX_LIMIT = 16;

const char* GlobalWriteFileNameFormat;
const char* GlobalImageListFolder;
std::string inFullStr;
std::string outFullStr;

inline void InitVideoReader(cv::VideoCapture& video_capture)
{
	std::string listNum = "2";
	// GlobalImageListFolder = "D:\\Bag\\Code_VS15\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";

	std::string inPrefix = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_";
	std::string inBackend = "_8bit\\Frame_%04d.png";
	inFullStr = inPrefix + listNum + inBackend;

	GlobalImageListFolder = inFullStr.c_str();

	std::string outPrefix = ".\\ir_file_20170531_1000m_";
	std::string outBackend = "\\Frame_%04d.png";
	outFullStr = outPrefix + listNum + outBackend;

	GlobalWriteFileNameFormat = outFullStr.c_str();

	video_capture.open(GlobalImageListFolder);
}
