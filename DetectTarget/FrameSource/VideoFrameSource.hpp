#pragma once
#include <string>

#include "CaptureFrameSource.hpp"

class VideoFrameSource : public CaptureFrameSource
{
public:
	explicit VideoFrameSource(const std::string& videoFileName);
	void reset() override;

private:
	std::string videoFileName;
};

inline VideoFrameSource::VideoFrameSource(const std::string& videoFileName) : videoFileName(videoFileName)
{
	VideoFrameSource::reset();
}

inline void VideoFrameSource::reset()
{
	videoCapture.release();
	videoCapture.open(videoFileName);
	CV_Assert(videoCapture.isOpened());
}