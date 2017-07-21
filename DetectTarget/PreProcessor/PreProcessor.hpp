#pragma once
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
#include <stack>
#include "../Utils/Util.hpp"
#include "../Models/DifferenceElem.hpp"

template <typename DataType>
class PreProcessor
{
public:
	PreProcessor();

	PreProcessor(int image_width, int image_height);

	void InitParameters();

	void SetSourceFrame(cv::Mat& frame);

	void SetBlockSize(int blockSize);

	void SetLowContrastThreshold(const int& lowContrastThreshold);

private:
	std::vector<std::vector<DataType>> GetMaxMinPixelValueDifferenceMap();

	std::vector<DifferenceElem> GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff);

	void GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec);

	void SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal);

	void UpdateConfigure();

	void AssertGray();

public:

	void Dilate(cv::Mat& resultFrame) const;

	void TopHat(cv::Mat& resultFrame);

	void Discrelize(cv::Mat& resultFrame);

	void Smooth(cv::Mat& resultFrame);

	void SetDilationKernelSize(const int& kernelSize);

	void StrengthenIntensityOfBlock();

private:
	int imageWidth;
	int imageHeight;

	int dilateKernelSize;
	int smoothKernelSize;

	int discrelizeStep;

	cv::Mat sourceFrame;

	int CountY;
	int CountX;
	int BlockSize;
	int LowContrastThreshold;
};

template <typename DataType>
PreProcessor<DataType>::PreProcessor()
	: imageWidth(0),
	imageHeight(0),
	dilateKernelSize(0),
	smoothKernelSize(0),
	discrelizeStep(0),
	CountY(0),
	CountX(0),
	BlockSize(10),
	LowContrastThreshold(3)
{
	InitParameters();
}

template <typename DataType>
PreProcessor<DataType>::PreProcessor(int image_width, int image_height)
	: imageWidth(image_width),
	imageHeight(image_height),
	dilateKernelSize(0),
	smoothKernelSize(0),
	discrelizeStep(0),
	CountY(0),
	CountX(0),
	BlockSize(10),
	LowContrastThreshold(3)
{
	InitParameters();
}

template <typename DataType>
void PreProcessor<DataType>::InitParameters()
{
	CountX = ceil(static_cast<double>(imageWidth) / BlockSize);
	CountY = ceil(static_cast<double>(imageHeight) / BlockSize);
}

template <typename DataType>
void PreProcessor<DataType>::SetSourceFrame(cv::Mat& frame)
{
	sourceFrame = frame;
	AssertGray();
}

template <typename DataType>
void PreProcessor<DataType>::SetBlockSize(int blockSize)
{
	BlockSize = blockSize;
	UpdateConfigure();
}

template <typename DataType>
void PreProcessor<DataType>::SetLowContrastThreshold(const int& lowContrastThreshold)
{
	LowContrastThreshold = lowContrastThreshold;
}

template <typename DataType>
std::vector<std::vector<DataType>> PreProcessor<DataType>::GetMaxMinPixelValueDifferenceMap()
{
	std::vector<std::vector<DataType>> maxmindiff(CountY, std::vector<DataType>(CountX, static_cast<DataType>(0)));
	for (auto br = 0; br < CountY; ++br)
	{
		auto height = br == (CountY - 1) ? imageHeight - (CountY - 1) * BlockSize : BlockSize;
		for (auto bc = 0; bc < CountX; ++bc)
		{
			auto width = bc == (CountX - 1) ? imageWidth - (CountX - 1) * BlockSize : BlockSize;
			maxmindiff[br][bc] =
				Util<DataType>::GetMaxValueOfBlock(sourceFrame(cv::Rect(bc * BlockSize, br * BlockSize, width, height))) -
				Util<DataType>::GetMinValueOfBlock(sourceFrame(cv::Rect(bc * BlockSize, br * BlockSize, width, height)));
		}
	}
	return maxmindiff;
}

template <typename DataType>
std::vector<DifferenceElem> PreProcessor<DataType>::GetMostMaxDiffBlock(std::vector<std::vector<DataType>> maxmindiff)
{
	std::vector<DifferenceElem> mostPossibleBlocks;

	GetDiffValueOfMatrixBigThanThreshold(maxmindiff, mostPossibleBlocks);

	return mostPossibleBlocks;
}

template <typename DataType>
void PreProcessor<DataType>::GetDiffValueOfMatrixBigThanThreshold(std::vector<std::vector<DataType>> maxmindiff, std::vector<DifferenceElem>& diffElemVec)
{
	std::vector<std::vector<bool>> flag(CountY, std::vector<bool>(CountX, false));
	diffElemVec.clear();
	for (auto br = 0; br < CountY; ++br)
	{
		for (auto bc = 0; bc < CountX; ++bc)
		{
			if (LowContrastThreshold <= static_cast<int>(maxmindiff[br][bc]))
			{
				DifferenceElem diffElem;
				diffElem.blockX = bc;
				diffElem.blockY = br;
				diffElem.diffVal = static_cast<int>(maxmindiff[br][bc]);
				diffElemVec.push_back(diffElem);

				flag[br][bc] = true;

				SearchNeighbors(maxmindiff, diffElemVec, flag, br, bc, static_cast<int>(maxmindiff[br][bc]));
			}
		}
	}
}

template <typename DataType>
void PreProcessor<DataType>::SearchNeighbors(const std::vector<std::vector<DataType>>& maxmindiff, std::vector<DifferenceElem>& diffElemVec, std::vector<std::vector<bool>>& flag, int br, int bc, int diffVal)
{
	auto threshold = 2;

	std::stack<cv::Point> deepTrace;
	deepTrace.push(cv::Point(bc, br));

	while (deepTrace.empty() != true)
	{
		auto top = deepTrace.top();
		deepTrace.pop();

		auto c = top.x;
		auto r = top.y;

		if (r - 1 >= 0 && flag[r - 1][c] == false && abs(static_cast<int>(maxmindiff[r - 1][c]) - diffVal) < threshold)
		{
			flag[r - 1][c] = true;
			deepTrace.push(cv::Point(c, r - 1));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r - 1][c];
			elem.blockX = c;
			elem.blockY = r - 1;
			diffElemVec.push_back(elem);
		}
		if (r + 1 < CountY && flag[r + 1][c] == false && abs(static_cast<int>(maxmindiff[r + 1][c]) - diffVal) < threshold)
		{
			flag[r + 1][c] = true;
			deepTrace.push(cv::Point(c, r + 1));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r + 1][c];
			elem.blockX = c;
			elem.blockY = r + 1;
			diffElemVec.push_back(elem);
		}
		if (c - 1 >= 0 && flag[r][c - 1] == false && abs(static_cast<int>(maxmindiff[r][c - 1]) - diffVal) < threshold)
		{
			flag[r][c - 1] = true;
			deepTrace.push(cv::Point(c - 1, r));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r][c - 1];
			elem.blockX = c - 1;
			elem.blockY = r;
			diffElemVec.push_back(elem);
		}
		if (c + 1 < CountX && flag[r][c + 1] == false && abs(static_cast<int>(maxmindiff[r][c + 1]) - diffVal) < threshold)
		{
			flag[r][c + 1] = true;
			deepTrace.push(cv::Point(c + 1, r));
			DifferenceElem elem;
			elem.diffVal = maxmindiff[r][c + 1];
			elem.blockX = c + 1;
			elem.blockY = r;
			diffElemVec.push_back(elem);
		}
	}
}

template <typename DataType>
void PreProcessor<DataType>::UpdateConfigure()
{
	InitParameters();
}

template <typename DataType>
void PreProcessor<DataType>::AssertGray()
{
	if (sourceFrame.channels() != 1)
	{
		std::cout << "Error => The input image must be gray!" << std::endl;
		sourceFrame = cv::Mat();
	}
}

template <typename DataType>
void PreProcessor<DataType>::Dilate(cv::Mat& resultFrame) const
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
void PreProcessor<DataType>::SetDilationKernelSize(const int& kernelSize)
{
	this->dilateKernelSize = kernelSize;
}

template <typename DataType>
void PreProcessor<DataType>::StrengthenIntensityOfBlock()
{
	auto maxmindiffMatrix = GetMaxMinPixelValueDifferenceMap();

	auto differenceElems = GetMostMaxDiffBlock(maxmindiffMatrix);

	for (auto elem : differenceElems)
	{
		auto centerX = elem.blockX * BlockSize + BlockSize / 2;
		auto centerY = elem.blockY * BlockSize + BlockSize / 2;
		auto boundingBoxLeftTopX = centerX - BlockSize >= 0 ? centerX - BlockSize : 0;
		auto boundingBoxLeftTopY = centerY - BlockSize >= 0 ? centerY - BlockSize : 0;
		auto boundingBoxRightBottomX = centerX + BlockSize < imageWidth ? centerX + BlockSize : imageWidth - 1;
		auto boundingBoxRightBottomY = centerY + BlockSize < imageHeight ? centerY + BlockSize : imageHeight - 1;

		auto averageValue = Util<DataType>::CalculateAverageValue(sourceFrame, boundingBoxLeftTopX, boundingBoxLeftTopY, boundingBoxRightBottomX, boundingBoxRightBottomY);

		auto maxdiffBlockRightBottomX = (elem.blockX + 1) * BlockSize > imageWidth ? imageWidth - 1 : (elem.blockX + 1) * BlockSize;
		auto maxdiffBlockRightBottomY = (elem.blockY + 1) * BlockSize > imageHeight ? imageHeight - 1 : (elem.blockY + 1) * BlockSize;

		for (auto r = elem.blockY * BlockSize; r < maxdiffBlockRightBottomY; ++r)
		{
			auto ptr = sourceFrame.ptr<DataType>(r);
			for (auto c = elem.blockX * BlockSize; c < maxdiffBlockRightBottomX; ++c)
			{
				if (ptr[c] > averageValue)
				{
					ptr[c] = ptr[c] + 10 > 255 ? 255 : ptr[c] + 10;
				}
			}
		}
	}
}
