#pragma once
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>

template <typename DataType>
class PreProcessor
{
public:
	PreProcessor();

	PreProcessor(unsigned int image_width, unsigned int image_height);

	void InitParameters();

	void SetSourceFrame(const cv::Mat& frame);

public:

	void Dilate(cv::Mat& resultFrame);

	void TopHat(cv::Mat& resultFrame);

	void Discrelize(cv::Mat& resultFrame);

	void Smooth(cv::Mat& resultFrame);

	void SetDilationKernelSize(const unsigned int& kernelSize);

private:
	unsigned int imageWidth;
	unsigned int imageHeight;

	unsigned int dilateKernelSize;
	unsigned int smoothKernelSize;

	unsigned int discrelizeStep;

	cv::Mat sourceFrame;
};

template <typename DataType>
PreProcessor<DataType>::PreProcessor() : imageWidth(0), imageHeight(0), dilateKernelSize(0), smoothKernelSize(0), discrelizeStep(0)
{
}

template <typename DataType>
PreProcessor<DataType>::PreProcessor(unsigned image_width, unsigned image_height) : imageWidth(image_width), imageHeight(image_height), dilateKernelSize(0), smoothKernelSize(0), discrelizeStep(0)
{
}

template <typename DataType>
void PreProcessor<DataType>::InitParameters()
{
}

template <typename DataType>
void PreProcessor<DataType>::SetSourceFrame(const cv::Mat& frame)
{
	if (frame.channels() == 3)
	{
		cvtColor(frame, this->sourceFrame, CV_RGB2GRAY);
	}
	else
	{
		sourceFrame = frame;
	}
}

template <typename DataType>
void PreProcessor<DataType>::Dilate(cv::Mat& resultFrame)
{
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(dilateKernelSize, dilateKernelSize));
	dilate(sourceFrame, resultFrame, kernel);
}

template <typename DataType>
void PreProcessor<DataType>::TopHat(cv::Mat& resultFrame)
{
}

template <typename DataType>
void PreProcessor<DataType>::Discrelize(cv::Mat& resultFrame)
{
}

template <typename DataType>
void PreProcessor<DataType>::Smooth(cv::Mat& resultFrame)
{
}

template <typename DataType>
void PreProcessor<DataType>::SetDilationKernelSize(const unsigned int& kernelSize)
{
	this->dilateKernelSize = kernelSize;
}
