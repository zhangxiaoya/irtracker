#pragma once
#include "FrameSource.hpp"
#include "ImageListReader/ImageListReader.hpp"
#include "../Headers/GlobalConstantConfigure.h"

class ImageListFrameSource :public FrameSource
{
public:
	explicit ImageListFrameSource(std::string file_name_format, int start_index = 0);

	void nextFrame(cv::OutputArray frame) override;

	void reset() override;

private:
	int currentIndex;
	int startIndex;
	std::string fileNameFormat;
	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];
};

inline ImageListFrameSource::ImageListFrameSource(std::string file_name_format, int start_index): currentIndex(0), startIndex(start_index), fileNameFormat(file_name_format)
{
}

inline void ImageListFrameSource::nextFrame(cv::OutputArray frame)
{
	sprintf_s(imageFullName, WRITE_FILE_NAME_BUFFER_SIZE, fileNameFormat.c_str(), currentIndex);
	auto temFrame = cv::imread(imageFullName);

	temFrame.copyTo(frame);
	++currentIndex;
}

inline void ImageListFrameSource::reset()
{
	currentIndex = 0;
}
