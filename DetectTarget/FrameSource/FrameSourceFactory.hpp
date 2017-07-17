#pragma once

#include "EmptyFrameSource.hpp"
#include "VideoFrameSource.hpp"
#include "ImageListFrameSource.hpp"

class FrameSourceFactory
{
public:
	static cv::Ptr<FrameSource> createEmptyFrameSource()
	{
		return new EmptyFrameSource();
	}

	static cv::Ptr<FrameSource> createFrameSourceFromVideo(const std::string& videoFileName)
	{
		return new VideoFrameSource(videoFileName);
	}

	static cv::Ptr<FrameSource> createFrameSourceFromImageList(std::string file_name_format, int start_index = 0)
	{
		return new ImageListFrameSource(file_name_format, start_index);
	}
};
