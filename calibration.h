#pragma once

#include <string>
#include <opencv2/opencv.hpp>


bool detectCorners(const cv::Mat& img, cv::Size boardSize, cv::Size winSize, std::vector<cv::Point2f>& corners);

void detectCorners(const std::vector<std::string>& imagePaths, cv::Size boardSize, cv::Size winSize, std::vector<std::vector<cv::Point2f>>& corners);

void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners);

bool calibrateCamera(const std::vector<std::vector<cv::Point2f>>& imgPoints, cv::Size imageSize, std::vector<cv::Point3f>& objPoints,
    cv::Mat& cameraMatrix, cv::Mat& distCoeffs, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs );

