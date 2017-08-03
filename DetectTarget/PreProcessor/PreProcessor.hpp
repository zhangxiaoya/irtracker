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

	double GetAverageGrayValueOfKNeighbor(int row, int col, int radius);

	void CalculateEntropy(double* entropy, int* frequency) const;

	void CalculatePixelFrequency(int* frequency);

	void GetMultiscalLocalDifferenceContrastMap(cv::Mat& multiscaleLocalContrastMap);

	void GetPixelFrequency(double* entropy);

	void GetUniquePixelValueOfKNeighbor(int row, int col, int radius, int* uniquePixelList);

	void GetLocalEntrogy(cv::Mat& localEntrogyMap);

public:
	void Dilate(cv::Mat& resultFrame) const;

	void TopHat(cv::Mat& resultFrame);

	void Discrelize(cv::Mat& resultFrame);

	void Smooth(cv::Mat& resultFrame);

	void SetDilationKernelSize(const int& kernelSize);

	void StrengthenIntensityOfBlock();

	void MultiScaleDifference(cv::Mat& resultFrame);

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
	unsigned int BitCount;
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
	BitCount = 1 << (8 * sizeof(DataType));
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
double PreProcessor<DataType>::GetAverageGrayValueOfKNeighbor(int row, int col, int radius)
{
	auto leftTopX = col - radius >= 0 ? col - radius : 0;
	auto leftTopY = row - radius >= 0 ? row - radius : 0;

	auto rightBottomX = col + radius < imageWidth ? col + radius : imageWidth - 1;
	auto rightBottomY = row + radius < imageHeight ? row + radius : imageHeight - 1;

	auto sum = 0;
	auto totalCount = 0;

	for (auto r = leftTopY; r <= rightBottomY; ++r)
	{
		auto ptr = sourceFrame.ptr<DataType>(r);
		for (auto c = leftTopX; c <= rightBottomX; ++c)
		{
			sum += static_cast<int>(ptr[c]);
			++totalCount;
		}
	}

	return static_cast<double>(sum / totalCount);
}

template <typename DataType>
void PreProcessor<DataType>::CalculateEntropy(double* entropy, int* frequency) const
{
	for (auto i = 0; i < BitCount; ++i)
	{
		auto probability = static_cast<double>(frequency[i]) / (imageHeight * imageWidth);
		entropy[i] = probability * log2(probability);
	}
}

template <typename DataType>
void PreProcessor<DataType>::CalculatePixelFrequency(int* frequency)
{
	for (auto r = 0; r < imageHeight; ++r)
	{
		auto ptr = sourceFrame.ptr<DataType>(r);
		for (auto c = 0; c < imageWidth; ++c)
		{
			frequency[static_cast<int>(ptr[c])] ++;
		}
	}
}

template <typename DataType>
void PreProcessor<DataType>::GetMultiscalLocalDifferenceContrastMap(cv::Mat& multiscaleLocalContrastMap)
{
	const auto L = 6;

	double averageOfKNeighbor[L] = { 0.0 };
	double contrastOfKNeighbor[L] = { 0.0 };

	for (auto r = 0; r < imageHeight; ++r)
	{
		for (auto c = 0; c < imageWidth; ++c)
		{
			memset(averageOfKNeighbor, 0, sizeof(double)*L);
			memset(contrastOfKNeighbor, 0, sizeof(double)*L);

			for (auto k = 1; k <= L; ++k)
			{
				averageOfKNeighbor[k - 1] = GetAverageGrayValueOfKNeighbor(r, c, k);
			}

			auto maxVal = Util<DataType>::MaxOfConstLengthList(averageOfKNeighbor, L);
			auto minVal = Util<DataType>::MinOfConstLengthList(averageOfKNeighbor, L);

			auto squareDiff = (maxVal - minVal) * (maxVal - minVal);

			if (squareDiff - 0.0 <= MinDiff || 0.0 - squareDiff <= MinDiff)
			{
				multiscaleLocalContrastMap.at<float>(r, c) = 1.0;
				//				logPrinter.PrintLogs("NAN", LogLevel::Waring);
				continue;
			}

			for (auto i = 0; i < L - 1; ++i)
			{
				contrastOfKNeighbor[i] = ((averageOfKNeighbor[i] - averageOfKNeighbor[L - 1]) * (averageOfKNeighbor[i] - averageOfKNeighbor[L - 1]) / squareDiff);
			}

			contrastOfKNeighbor[L - 1] = 0.0;

			multiscaleLocalContrastMap.at<float>(r, c) = Util<DataType>::MaxOfConstLengthList(contrastOfKNeighbor, L);
		}
	}
}

template <typename DataType>
void PreProcessor<DataType>::GetPixelFrequency(double* entropy)
{
	auto frequency = new int[BitCount];
	memset(frequency, 0, sizeof(int) * BitCount);

	CalculatePixelFrequency(frequency);

	CalculateEntropy(entropy, frequency);

	delete[] frequency;
}

template <typename DataType>
void PreProcessor<DataType>::GetUniquePixelValueOfKNeighbor(int row, int col, int radius, int* uniquePixelList)
{
	memset(uniquePixelList, 0, sizeof(int) * 256);

	auto leftTopX = col - radius >= 0 ? col - radius : 0;
	auto leftTopY = row - radius >= 0 ? row - radius : 0;

	auto rightBottomX = col + radius < imageWidth ? col + radius : imageWidth - 1;
	auto rightBottomY = row + radius < imageHeight ? row + radius : imageHeight - 1;

	for (auto r = leftTopY; r <= rightBottomY; ++r)
	{
		auto ptr = sourceFrame.ptr<DataType>(r);
		for (auto c = leftTopX; c <= rightBottomX; ++c)
		{
			uniquePixelList[static_cast<int>(ptr[c])] ++;
		}
	}
}

template <typename DataType>
void PreProcessor<DataType>::GetLocalEntrogy(cv::Mat& localEntrogyMap)
{
	auto entropy = new double[BitCount];
	auto uniquePixelList = new int[BitCount];

	memset(entropy, 0, sizeof(double)*BitCount);
	memset(uniquePixelList, 0, sizeof(int) * BitCount);

	GetPixelFrequency(entropy);

	for (auto r = 0; r < imageHeight; ++r)
	{
		auto ptrDst = localEntrogyMap.ptr<float>(r);
		auto ptrSrc = sourceFrame.ptr<DataType>(r);

		for (auto c = 0; c < imageWidth; ++c)
		{
			GetUniquePixelValueOfKNeighbor(r, c, 2, uniquePixelList);

			auto w = 0.0;
			for (auto i = 0; i < 256; ++i)
			{
				if (uniquePixelList[i] != 0)
				{
					w += static_cast<double>((i - static_cast<int>(ptrSrc[c])) * (i - static_cast<int>(ptrSrc[c]))) * entropy[i];
				}
			}
			ptrDst[c] = -1 * w;
		}
	}

	delete[] entropy;
	delete[] uniquePixelList;
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

template <typename DataType>
void PreProcessor<DataType>::MultiScaleDifference(cv::Mat& resultFrame)
{
	cv::Mat multiscaleLocalContrastMap(cv::Size(imageWidth, imageHeight), CV_32FC1, cv::Scalar(0));

	GetMultiscalLocalDifferenceContrastMap(multiscaleLocalContrastMap);

	cv::Mat localEntrogyMap(cv::Size(imageWidth, imageHeight), CV_32FC1, cv::Scalar(0));

	GetLocalEntrogy(localEntrogyMap);

	resultFrame = localEntrogyMap.mul(multiscaleLocalContrastMap);
}
