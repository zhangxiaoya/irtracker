#pragma once
#include "FrameSource.hpp"

class EmptyFrameSource : public FrameSource
{
public:
	void nextFrame(cv::OutputArray frame) override;
	void reset() override;
};

inline void EmptyFrameSource::nextFrame(cv::OutputArray frame)
{
	frame.release();
}

inline void EmptyFrameSource::reset()
{

}