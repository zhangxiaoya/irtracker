#pragma once
#include <core/core.hpp>
#include <highgui/highgui.hpp>

#include "GlobalInitialUtil.hpp"
#include <iomanip>

class ToVideo
{
public:
	explicit ToVideo(const std::string save_file_folder, int ftps = 5, const std::string& name_template = "Frame_%08", const std::string& save_file_name = "video")
		: ftps(ftps),
		  nameTemplate(name_template),
		  saveFileName(save_file_name),
		  saveFileFolder(save_file_folder)
	{
	}

	void PutAllResultFramesToOneVideo() const;

	void SetFrameSize(int imageWidth, int imageHeight);

private:
	int ftps;
	std::string nameTemplate;
	std::string saveFileName;
	std::string saveFileFolder;
	cv::Size frameSize;
};

inline void ToVideo::SetFrameSize(int imageWidth, int imageHeight)
{
	frameSize = cv::Size(imageWidth, imageHeight);
}

inline void ToVideo::PutAllResultFramesToOneVideo() const
{
	auto saveFileFullName = saveFileFolder + "\\" + saveFileName + ".avi";

	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];

	cv::VideoWriter video_writer;
	video_writer.open(saveFileFullName, -1, ftps, frameSize);

	if (video_writer.isOpened())
	{
		auto index = 1;
		while (true)
		{
			sprintf_s(imageFullName, WRITE_FILE_NAME_BUFFER_SIZE, GlobalWriteFileNameFormat, index++);
			auto img = cv::imread(imageFullName);

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
