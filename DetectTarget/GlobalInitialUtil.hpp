#pragma once

const char* GlobalWriteFileNameFormat;
const char* GlobalImageListFolder;

inline void InitVideoReader(cv::VideoCapture& video_capture)
{

	// GlobalImageListFolder = "D:\\Bag\\Code_VS15\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit_maxFilter_discrezated\\Frame_%04d.png";

	GlobalImageListFolder = "E:\\WorkLogs\\Gitlab\\ExtractVideo\\ExtractVideo\\ir_file_20170531_1000m_2_8bit\\Frame_%04d.png";

	GlobalWriteFileNameFormat = ".\\ir_file_20170531_1000m_2\\Frame_%04d.png";

	video_capture.open(GlobalImageListFolder);
}
