#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

int main(int argc, char* argv[])
{
	InitGlobalConfigure();

	auto frameSource = FrameSourceFactory::createFrameSourceFromImageList(GlobalImageListNameFormat, 0);

	auto framePersistance = FramePersistanceFactory::createFramePersistance(GlobalWriteFileNameFormat);

	auto monitor = MonitorFactory::CreateMonitor<uchar>(frameSource, framePersistance);

	if (ImageListReadFlag == true)
	{
		logPrinter.PrintLogs("Open Image List Success!", LogLevel::Info);

		monitor->SetResultPersistanceFlag(true, true);

		monitor->Process();
	}
	else
	{
		logPrinter.PrintLogs("Open Image List Failed", LogLevel::Error);
	}

	Cleaner();
	return 0;
}
