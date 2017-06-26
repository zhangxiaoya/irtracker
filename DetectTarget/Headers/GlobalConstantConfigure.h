#pragma once
#include <opencv2/core/core.hpp>

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
const auto TARGET_WIDTH_MAX_LIMIT = 16;
const auto TARGET_HEIGHT_MAX_LIMIT = 16;

static int IMAGE_WIDTH = 320;
static int IMAGE_HEIGHT = 256;

static auto countX = ceil(static_cast<double>(IMAGE_WIDTH) / BLOCK_SIZE);
static auto countY = ceil(static_cast<double>(IMAGE_HEIGHT) / BLOCK_SIZE);

const auto WRITE_FILE_NAME_BUFFER_SIZE = 200;

static auto DISCRATED_BIN = 25;

static auto DATA_TYPE = CV_8UC1;
