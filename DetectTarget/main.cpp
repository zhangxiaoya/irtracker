#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "Utils/Util.hpp"
#include "FrameSource/FrameSourceFactory.hpp"
#include "FramePersistance/FramePersistanceFactory.hpp"
#include "Monitor/Monitor.hpp"
#include "Monitor/MonitorFactory.hpp"

void Cleaner()
{
	destroyAllWindows();
	system("pause");
}

void PrintLogs(string text)
{
	std::cout << text << std::endl;
}

int main(int argc, char* argv[])
{
	InitGlobalConfigure();

	auto frameSource = FrameSourceFactory::createFrameSourceFromImageList(GlobalImageListNameFormat, 0);

	auto framePersistance = FramePersistanceFactory::createFramePersistance(GlobalWriteFileNameFormat);

	auto monitor = MonitorFactory::CreateMonitor<uchar>(frameSource, framePersistance);

	if (ImageListReadFlag == true)
	{
		PrintLogs("Open Image List Success!");

		monitor->Process();
	}
	else
	{
		PrintLogs("Open Image List Failed");
	}

	Cleaner();
	return 0;
}
