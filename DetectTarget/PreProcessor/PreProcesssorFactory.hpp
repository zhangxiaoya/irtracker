#pragma once
#include <core/core.hpp>
#include "PreProcessor.hpp"

class PreProcessorFactory
{
public:
	template<typename DataType>
	static cv::Ptr<PreProcessor<DataType>> CreatePreProcessor(const int& imageWidth, const int& imageHeight);
};

template <typename DataType>
cv::Ptr<PreProcessor<DataType>> PreProcessorFactory::CreatePreProcessor(const int& imageWidth, const int& imageHeight)
{
	return new PreProcessor<DataType>(imageWidth,imageHeight);
}
