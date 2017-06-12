#pragma once
#include "Util.hpp"

inline void InitVideoReader(cv::VideoCapture& video_capture)
{
	const char* firstImageList;

	if (AFTER_MAX_FILTER)
	{
		//		firstImageList = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_1_8bit_maxFilter_discrezated\\Frame_%04d.png";
		//		firstImageList = "D:\\Bag\\Code_VS15\\Data\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";
		//				firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit_maxFilter_discrezated\\Frame_%04d.png";
		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";
	}
	else
	{
		// firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";
		firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit\\Frame_%04d.png";
	}

	video_capture.open(firstImageList);
}
