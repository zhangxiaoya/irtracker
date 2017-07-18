#pragma once
#include "Monitor.hpp"

class MonitorFactory
{
public:
	static cv::Ptr<Monitor> CreateMonitor(cv::Ptr<FrameSource> frameSource, cv::Ptr<FramePersistance> framePersistance);
};

inline cv::Ptr<Monitor> MonitorFactory::CreateMonitor(cv::Ptr<FrameSource> frameSource, cv::Ptr<FramePersistance> framePersistance)
{
	return new Monitor(frameSource, framePersistance);
}
