#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

/*
Given a grayscale image, outputs a binary image containing the detected edges.
*/
void detectEdges(const cv::Mat& src_gray, cv::Mat& detected_edges);

void findAllContours(const cv::Mat& edges, std::vector<std::vector<cv::Point2i>>& contours);

void findLargestContour(const std::vector<std::vector<cv::Point2i>>& contours, std::vector<cv::Point2i>& contour);

void findLargestContour(const cv::Mat& edges, std::vector<cv::Point2i>& contour);

void detectLines(const cv::Mat& edges, std::vector<cv::Vec4i>& lines);

void reduceLines(std::vector<cv::Vec4i>& lines);

void discardExternalLines(const std::vector<cv::Point2i>& contour, std::vector<cv::Vec4i>& lines, cv::Size imageSize);

template <typename _Tp> 
bool isInside(const cv::Size& imgSize, const _Tp& point);

std::pair<cv::Point2i, cv::Point2i> lineToPoints(const cv::Vec4i line);

bool intersectLines(cv::Vec4i line1, cv::Vec4i line2, cv::Point2f& intersection);

std::vector<cv::Point2f> findIntersections(const std::vector<cv::Vec4i>& lines, double minAngle);

std::vector<cv::Point2f> findIntersections(const std::vector<cv::Vec4i>& lines, cv::Size imageSize);


template<typename _Tp>
void clipPoints(std::vector<_Tp>& points, cv::Size clipWindow);

template<typename _Tp>
bool isCloseToOtherPoints(const _Tp& point, const std::vector<_Tp>& points, double thresholdDistance);

template<typename _Tp>
void filterClosePoints(std::vector<_Tp>& points, double thresholdDistance);

template<typename _Tp>
void reduceConvexHull(std::vector<_Tp>& convexHull, size_t max_size);