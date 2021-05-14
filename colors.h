#pragma once

#include <opencv2/opencv.hpp>

cv::Vec3b convHSVToRGB(float H, float S, float V);

uchar genRandomGrayColor();

cv::Vec3b genRandomBGRColor();

// based on https://stackoverflow.com/a/1168291
cv::Vec3b genUniqueBGRColor();
