#pragma once
#include "../Utils/Util.hpp"
#include "../Utils/SpecialUtil.hpp"
#include "../Detector/DetectByMaxFilterAndAdptiveThreshold.hpp"
#include <iomanip>
#include "../Models/DrawResultType.hpp"
#include "../FramePersistance/FramePersistance.hpp"

class Monitor
{
public:
	Monitor(Ptr<FrameSource> frameSource, Ptr<FramePersistance> framePersistance);

	void Process();

protected:
	bool CheckOriginalImageSuroundedBox(const cv::Mat& grayFrame, const cv::Rect& rect) const;

	bool CheckDecreatizatedImageSuroundedBox(const cv::Mat& fdImg, const struct CvRect& rect) const;

	bool CheckFourBlock(const cv::Mat& fdImg, const cv::Rect& rect) const;

private:
	void DrawResult(cv::Mat& colorFrame, const cv::Rect& rect, DrawResultType drawResultType = DrawResultType::Rectangles) const;

	static void DrawHalfRectangle(cv::Mat& colorFrame, const int left, const int top, const int right, const int bottom, const cv::Scalar& lineColor);

	cv::Mat curFrame;
	cv::Mat grayFrame;
	cv::Mat colorFrame;

	cv::Mat preprocessResultFrame;
	cv::Mat detectedResultFrame;

	int frameIndex;
	
	cv::Ptr<FrameSource> frameSource;
	cv::Ptr<FramePersistance> framePersistance;
};

inline Monitor::Monitor(cv::Ptr<FrameSource> frameSource, cv::Ptr<FramePersistance> framePersistance): frameIndex(0)
{
	this->framePersistance = framePersistance;
	this->frameSource = frameSource;
}

inline void Monitor::Process()
{
	while (!curFrame.empty() || frameIndex == 0)
	{
		frameSource->nextFrame(curFrame);

		if (curFrame.empty() != true)
		{
			Util::ShowImage(curFrame);

			if (SpecialUtil::CheckFrameIsGray(curFrame, grayFrame))
			{
				cvtColor(curFrame, colorFrame, CV_GRAY2BGR);
			}
			else
			{
				colorFrame = curFrame;
			}

			auto targetRects = DetectByMaxFilterAndAdptiveThreshold::Detect<uchar>(grayFrame,preprocessResultFrame, detectedResultFrame);

			for (auto rect : targetRects)
			{
				if (
					(
					(CHECK_ORIGIN_FLAG && CheckOriginalImageSuroundedBox(grayFrame, rect)) ||
						(CHECK_DECRETIZATED_FLAG && CheckDecreatizatedImageSuroundedBox(preprocessResultFrame, rect))
						)
					&&
					CheckFourBlock(preprocessResultFrame, rect)
					)
				{
					DrawResult(colorFrame, rect, DrawResultType::Rectangles);
				}
			}

			imshow("Last Detect and Tracking Result", colorFrame);

			framePersistance->Persistance(colorFrame);

			std::cout << "Index : " << std::setw(4) << frameIndex++ << std::endl;
		}
	}
}

inline bool Monitor::CheckOriginalImageSuroundedBox(const cv::Mat& grayFrame, const cv::Rect& rect) const
{
	auto centerX = rect.x + rect.width / 2;
	auto centerY = rect.y + rect.height / 2;

	auto surroundingBoxWidth = 3 * rect.width;
	auto surroundingBoxHeight = 3 * rect.height;

	auto boxLeftTopX = centerX - surroundingBoxWidth / 2 >= 0 ? centerX - surroundingBoxWidth / 2 : 0;
	auto boxLeftTopY = centerY - surroundingBoxHeight / 2 >= 0 ? centerY - surroundingBoxHeight / 2 : 0;
	auto boxRightBottomX = centerX + surroundingBoxWidth / 2 < IMAGE_WIDTH ? centerX + surroundingBoxWidth / 2 : IMAGE_WIDTH - 1;
	auto boxRightBottomY = centerY + surroundingBoxHeight / 2 < IMAGE_HEIGHT ? centerY + surroundingBoxHeight / 2 : IMAGE_HEIGHT - 1;

	auto avgValOfSurroundingBox = Util::AverageValue(grayFrame, cv::Rect(boxLeftTopX, boxLeftTopY, boxRightBottomX - boxLeftTopX + 1, boxRightBottomY - boxLeftTopY + 1));
	auto avgValOfCurrentRect = Util::AverageValue(grayFrame, rect);

	auto convexThreshold = avgValOfSurroundingBox + avgValOfSurroundingBox / 17;
	auto concaveThreshold = avgValOfSurroundingBox - avgValOfSurroundingBox / 20;

	if (std::abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < 3)
		return false;

	uchar centerValue = 0;
	Util::CalCulateCenterValue(grayFrame, centerValue, rect);

	if (avgValOfCurrentRect > convexThreshold || avgValOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

inline bool Monitor::CheckDecreatizatedImageSuroundedBox(const cv::Mat& fdImg, const struct CvRect& rect) const
{
	auto centerX = rect.x + rect.width / 2;
	auto centerY = rect.y + rect.height / 2;

	auto boxLeftTopX = centerX - 2 * rect.width / 2 >= 0 ? centerX - 2 * rect.width / 2 : 0;
	auto boxLeftTopY = centerY - 2 * rect.height / 2 >= 0 ? centerY - 2 * rect.height / 2 : 0;
	auto boxRightBottomX = centerX + 2 * rect.width / 2 < IMAGE_WIDTH ? centerX + 2 * rect.width / 2 : IMAGE_WIDTH - 1;
	auto boxRightBottomY = centerY + 2 * rect.height / 2 < IMAGE_HEIGHT ? centerY + 2 * rect.height / 2 : IMAGE_HEIGHT - 1;

	auto avgValOfSurroundingBox = Util::AverageValue(fdImg, cv::Rect(boxLeftTopX, boxLeftTopY, boxRightBottomX - boxLeftTopX + 1, boxRightBottomY - boxLeftTopY + 1));
	auto avgValOfCurrentRect = Util::AverageValue(fdImg, rect);

	auto convexThreshold = avgValOfSurroundingBox + avgValOfSurroundingBox / 8;
	auto concaveThreshold = avgValOfSurroundingBox - avgValOfSurroundingBox / 10;

	if (std::abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < 3)
		return false;

	uchar centerValue = 0;
	Util::CalCulateCenterValue(fdImg, centerValue, rect);

	if (avgValOfCurrentRect > convexThreshold || avgValOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

inline bool Monitor::CheckFourBlock(const cv::Mat& fdImg, const cv::Rect& rect) const
{
	auto curBlockX = rect.x / BLOCK_SIZE;
	auto curBlockY = rect.y / BLOCK_SIZE;

	if (curBlockX - 1 < 0 || curBlockX + 1 > countX || curBlockY - 1 < 0 || curBlockY + 1 > countY)
		return false;

	auto upAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY - 1);
	auto downAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY + 1);

	auto leftAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY - 1);
	auto rightAvg = Util::CalculateAverageValueWithBlockIndex(fdImg, curBlockX, curBlockY + 1);

	if (abs(static_cast<int>(upAvg) - static_cast<int>(downAvg)) > 8)
		return false;

	if (abs(static_cast<int>(leftAvg) - static_cast<int>(rightAvg)) > 8)
		return false;

	return true;
}

inline void Monitor::DrawResult(cv::Mat& colorFrame, const cv::Rect& rect, DrawResultType drawResultType) const
{
	auto left = rect.x - 2 < 0 ? 0 : rect.x - 2;
	auto top = rect.y - 2 < 0 ? 0 : rect.y - 2;
	auto right = rect.x + rect.width + 1 >= IMAGE_WIDTH ? IMAGE_WIDTH - 1 : rect.x + rect.width + 1;
	auto bottom = rect.y + rect.height + 1 >= IMAGE_HEIGHT ? IMAGE_HEIGHT - 1 : rect.y + rect.height + 1;
	auto lineColor = COLOR_RED;

	switch (drawResultType)
	{
	case DrawResultType::Rectangles:
	{
		rectangle(colorFrame, cv::Rect(left, top, rect.width + 4, rect.height + 4), COLOR_RED);
		break;
	}

	case DrawResultType::HalfRectangle:
	{
		this->DrawHalfRectangle(colorFrame, left, top, right, bottom, lineColor);

		break;
	}
	case DrawResultType::Target:
	{
		line(colorFrame, cv::Point(left - 6, (top + bottom) / 2), cv::Point(left - 2, (top + bottom) / 2), lineColor, 1, CV_AA);
		line(colorFrame, cv::Point(right + 2, (top + bottom) / 2), cv::Point(right + 6, (top + bottom) / 2), lineColor, 1, CV_AA);

		line(colorFrame, cv::Point((left + right) / 2, top - 6), cv::Point((left + right) / 2, top - 2), lineColor, 1, CV_AA);
		line(colorFrame, cv::Point((left + right) / 2, bottom + 2), cv::Point((left + right) / 2, bottom + 6), lineColor, 1, CV_AA);
		break;
	}
	case DrawResultType::HalfRectangleWithLine:
	{
		this->DrawHalfRectangle(colorFrame, left, top, right, bottom, lineColor);

		line(colorFrame, cv::Point(0, (top + bottom) / 2), cv::Point(left - 2, (top + bottom) / 2), lineColor, 1, CV_AA);
		line(colorFrame, cv::Point(right + 2, (top + bottom) / 2), cv::Point(IMAGE_WIDTH - 1, (top + bottom) / 2), lineColor, 1, CV_AA);

		line(colorFrame, cv::Point((left + right) / 2, 0), cv::Point((left + right) / 2, top - 2), lineColor, 1, CV_AA);
		line(colorFrame, cv::Point((left + right) / 2, bottom + 2), cv::Point((left + right) / 2, IMAGE_HEIGHT - 1), lineColor, 1, CV_AA);
		break;
	}
	default:
		rectangle(colorFrame, cv::Rect(left, top, rect.width + 4, rect.height + 4), COLOR_RED);
		break;
	}
}

inline void Monitor::DrawHalfRectangle(cv::Mat& colorFrame, const int left, const int top, const int right, const int bottom, const cv::Scalar& lineColor)
{
	line(colorFrame, cv::Point(left, top), cv::Point(left, top + 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(left, top), cv::Point(left + 3, top), lineColor, 1, CV_AA);

	line(colorFrame, cv::Point(right, top), cv::Point(right, top + 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(right, top), cv::Point(right - 3, top), lineColor, 1, CV_AA);

	line(colorFrame, cv::Point(left, bottom), cv::Point(left, bottom - 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(left, bottom), cv::Point(left + 3, bottom), lineColor, 1, CV_AA);

	line(colorFrame, cv::Point(right, bottom), cv::Point(right, bottom - 3), lineColor, 1, CV_AA);
	line(colorFrame, cv::Point(right, bottom), cv::Point(right - 3, bottom), lineColor, 1, CV_AA);
}
