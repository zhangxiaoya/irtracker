#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "Utils/Util.hpp"
#include "FrameSource/FrameSourceFactory.hpp"
#include "FramePersistance/FramePersistanceFactory.hpp"
#include "Monitor/Monitor.hpp"
#include "Monitor/MonitorFactory.hpp"
#include "Models/LogLevel.hpp"

void Cleaner()
{
	destroyAllWindows();
	system("pause");
}

void PrintLogs(string text, LogLevel logLevel)
{
	switch (logLevel)
	{
	case LogLevel::Error:
		{
			std::cout << "ERROR => ";
			break;
		}
	case LogLevel::Info:
		{
			std::cout << "INFO => ";
			break;
		}
	case LogLevel::Waring:
		{
			std::cout << "WARNING => ";
			break;
		}
	default:
		{
			std::cout << "Not Defined => ";
			break;
		};
	}

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
		PrintLogs("Open Image List Success!", LogLevel::Info);

		monitor->Process();
	}
	else
	{
		PrintLogs("Open Image List Failed", LogLevel::Error);
	}

	Cleaner();
	return 0;
}
