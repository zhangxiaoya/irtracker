#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "Utils/Util.hpp"
#include "FrameSource/FrameSourceFactory.hpp"
#include "FramePersistance/FramePersistanceFactory.hpp"
#include "Monitor/Monitor.hpp"
#include "Monitor/MonitorFactory.hpp"

int main(int argc, char* argv[])
{
	InitConfigure();

	auto frameSource = FrameSourceFactory::createFrameSourceFromImageList(GlobalImageListNameFormat, 0);
	auto framePersistance = FramePersistanceFactory::createFramePersistance(GlobalWriteFileNameFormat);

	auto monitor = MonitorFactory::CreateMonitor(frameSource, framePersistance);

	if (ImageListReadFlag == true)
	{
		std::cout << "Open Image List Success!" << std::endl;

		monitor->Process();

		destroyAllWindows();
	}
	else
	{
		std::cout << "Open Image List Failed" << std::endl;
	}

	system("pause");
	return 0;
}
