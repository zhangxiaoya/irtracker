#pragma once
#include "Monitor.hpp"

class MonitorFactory
{
public:
	template<typename DataType>
	static cv::Ptr<Monitor<DataType>> CreateMonitor(cv::Ptr<FrameSource> frameSource, cv::Ptr<FramePersistance> framePersistance);
};

template <typename DataType>
cv::Ptr<Monitor<DataType>> MonitorFactory::CreateMonitor(cv::Ptr<FrameSource> frameSource, cv::Ptr<FramePersistance> framePersistance)
{
	return new Monitor<DataType>(frameSource, framePersistance);
}
