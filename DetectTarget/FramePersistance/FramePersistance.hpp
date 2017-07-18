#pragma once
#include "../Headers/GlobalConstantConfigure.h"
#include <highgui/highgui.hpp>

class FramePersistance
{
public:
	explicit FramePersistance(std::string fileNameFormat);

	void Persistance(cv::InputArray frame);

	void Reset();

private:
	int currentIndex;
	std::string fileNameFormat;
	char imageFullName[WRITE_FILE_NAME_BUFFER_SIZE];
};

inline FramePersistance::FramePersistance(std::string fileNameFormat): currentIndex(0),fileNameFormat(fileNameFormat)
{
}

inline void FramePersistance::Persistance(cv::InputArray frame)
{
	sprintf_s(imageFullName, fileNameFormat.c_str(), currentIndex);
	currentIndex++;

	imwrite(imageFullName, frame.getMat());
}

inline void FramePersistance::Reset()
{
	currentIndex = 0;
}
