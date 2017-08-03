#pragma once
#include <string>
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
#include <highgui/highgui.hpp>

#include "GlobalInitialUtil.hpp"

class ToVideo
{
public:
	explicit ToVideo(int ftps = 5, const std::string& name_template = "Frame_%08", const std::string& save_file_name = "video", const std::string& save_file_format = "png")
		: ftps(ftps),
		  nameTemplate(name_template),
		  saveFileName(save_file_name),
		  saveFileFormat(save_file_format)
	{
	}

	void PutAllResultFramesToOneVideo();

private:
	int ftps;
	std::string nameTemplate;
	std::string saveFileName;
	std::string saveFileFormat;
	cv::Size frameSize;
};

inline void ToVideo::PutAllResultFramesToOneVideo()
{
	auto len = strlen(GlobalWriteFileNameFormat);
	--len;
	for (; len >= 0; --len)
	{
		if(GlobalWriteFileNameFormat[len] =='\\')
			break;
	}

	auto videoFileFormatPrefix = new char[len-1];
	memcpy(videoFileFormatPrefix, GlobalWriteFileNameFormat, len);

	std::string videoFileFormat(videoFileFormatPrefix);
	videoFileFormat += "\\\\";
	videoFileFormat += saveFileName;
	videoFileFormat += ".avi";


	delete[] videoFileFormatPrefix;

	cv::VideoWriter video_writer;
		video_writer.open(saveFileName, -1, ftps, frameSize);

	if (video_writer.isOpened())
	{
		auto index = 1;
		while (true)
		{
			sprintf_s(fileName, buffer_count, imageListFormat.c_str(), index++);
			img = cv::imread(fileName);

			if (img.empty())
				break;

			video_writer << img;

			std::cout << "index : " << std::setw(4) << index << std::endl;
		}
		video_writer.release();
		std::cout << "Done!" << std::endl;
	}
	else
	{
		std::cout << "Save Video File Failed" << std::endl;
	}
}
