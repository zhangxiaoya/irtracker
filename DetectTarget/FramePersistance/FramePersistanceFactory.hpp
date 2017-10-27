#pragma once
#include "FramePersistance.hpp"

class FramePersistanceFactory
{
public:
	static cv::Ptr<FramePersistance> createFramePersistance(std::string imageFileNameFormat)
	{
		return new FramePersistance(imageFileNameFormat);
	}
};
