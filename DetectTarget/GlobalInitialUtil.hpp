#pragma once

inline void InitVideoReader(cv::VideoCapture& video_capture)
{
	// auto firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_1_8bit\\Frame_%04d.png";

	auto firstImageList = "D:\\Bag\\Code_VS15\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";
	// auto firstImageList = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit\\Frame_%04d.png";

	video_capture.open(firstImageList);
}
