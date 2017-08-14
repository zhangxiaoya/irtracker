#pragma once
#include <opencv2/core/core.hpp>
#include "../LogPrinter/LogPrinter.hpp"

auto const COLOR_RED = cv::Scalar(0, 0, 255);
auto const COLOR_BLUE = cv::Scalar(255, 0, 0);
auto const COLOR_GREEN = cv::Scalar(0, 255, 0);
auto const COLOR_YELLOW = cv::Scalar(0, 255, 255);

const auto BLOCK_SIZE = 10;
const auto TOP_COUNT_OF_BLOCK_WITH_HIGH_QUEUE_VALUE = 10;
const auto TOP_COUNT_OF_TARGET_WITH_HIGH_CONFIDENCE_VALUE = 5;

const auto QUEUE_SIZE = 5;

const auto SEARCH_WINDOW_WIDTH = 8;
const auto SEARCH_WINDOW_HEIGHT = 8;
const auto THRESHOLD = 25;
const auto SHOW_DELAY = 1;

const auto THINGKING_STAGE = 6;

const auto TARGET_WIDTH_MIN_LIMIT = 2;
const auto TARGET_HEIGHT_MIN_LIMIT = 2;
const auto TARGET_WIDTH_MAX_LIMIT = 30;
const auto TARGET_HEIGHT_MAX_LIMIT = 30;

static int IMAGE_WIDTH = 320;
static int IMAGE_HEIGHT = 256;

static auto countX = ceil(static_cast<double>(IMAGE_WIDTH) / BLOCK_SIZE);
static auto countY = ceil(static_cast<double>(IMAGE_HEIGHT) / BLOCK_SIZE);

const auto WRITE_FILE_NAME_BUFFER_SIZE = 200;

static auto DISCRATED_BIN = 15;

static auto CV_DATA_TYPE = CV_8UC1;

const auto DilateKernelSize = 3;

static bool CHECK_ORIGIN_FLAG = true;
static bool CHECK_DECRETIZATED_FLAG = true;
static bool CHECK_SURROUNDING_BOUNDARY_FLAG = true;
static bool CHECK_INSIDE_BOUNDARY_FLAG = true;
static bool CHECK_FOUR_BLOCK_FLAG = true;
static bool CHECK_COVERAGE_FLAG = true;

static auto ConvexPartitionOfOriginalImage = 0;
static auto ConcavePartitionOfOriginalImage = 0;

static auto ConvexPartitionOfDiscretizedImage = 0;
static auto ConcavePartitionOfDiscretizedImage = 0;

static auto IsNeedStrengthenIntensity = false;

const auto LowContrastThreshold = 3;

const auto MinDiffOfConvextAndConcaveThreshold = 3;

const double MinDiff = 0.00000001;

LogPrinter logPrinter;