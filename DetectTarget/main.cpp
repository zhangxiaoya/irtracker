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

int main(int argc, char* argv[])
{
	InitGlobalConfigure();

	auto frameSource = FrameSourceFactory::createFrameSourceFromImageList(GlobalImageListNameFormat, 0);

	auto framePersistance = FramePersistanceFactory::createFramePersistance(GlobalWriteFileNameFormat);

	auto monitor = MonitorFactory::CreateMonitor(frameSource, framePersistance);

	if (ImageListReadFlag == true)
	{
		std::cout << "Open Image List Success!" << std::endl;

		monitor->Process();
	}
	else
	{
		std::cout << "Open Image List Failed" << std::endl;
	}

	Cleaner();
	return 0;
}
