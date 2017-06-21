#pragma once
#include <opencv2/core/core.hpp>

auto const REDCOLOR = cv::Scalar(0, 0, 255);
auto const BLUECOLOR = cv::Scalar(255, 0, 0);
auto const GREENCOLOR = cv::Scalar(0, 255, 0);
auto const YELLOWCOLOR = cv::Scalar(0, 255, 255);

const auto STEP = 10;

const auto WINDOW_WIDTH = 8;
const auto WINDOW_HEIGHT = 8;
const auto THRESHHOLD = 25;
const auto SHOW_DELAY = 1;

const auto ThinkingTime = 6;

const auto TARGET_WIDTH_MIN_LIMIT = 2;
const auto TARGET_HEIGHT_MIN_LIMIT = 2;
const auto TARGET_WIDTH_MAX_LIMIT = 16;
const auto TARGET_HEIGHT_MAX_LIMIT = 16;

static int IMAGEWIDTH = 320;
static int IMAGEHEIGHT = 256;

static auto countX = ceil(static_cast<double>(IMAGEWIDTH) / STEP);
static auto countY = ceil(static_cast<double>(IMAGEWIDTH) / STEP);

const auto WRITE_FILE_NAME_BUFFER_SIZE = 200;