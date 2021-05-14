#include "chessboard.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#include "util.h"

using namespace cv;


void detectEdges(const Mat& src_gray, Mat& detected_edges) {
	const double lowThreshold = 50;
	const double thresholdRatio = 3;
	const double highThreshold = lowThreshold * thresholdRatio;
	const int kernelSize = 3;

	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, highThreshold, kernelSize);
}

void findAllContours(const Mat& edges, std::vector<std::vector<Point2i>>& contours) {
	findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);
}

void findLargestContour(const std::vector<std::vector<Point2i>>& contours, std::vector<Point2i>& contour) {
	assert(!contours.empty());

	int largest_index = -1;
	double largest_area = 0;

	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		const double area = contourArea(contours[i], false);  //  Find the area of contour
		if (area > largest_area) {
			largest_area = area;
			largest_index = i;                //Store the index of largest contour
		}
	}

	contour = contours[largest_index];
}

void findLargestContour(const Mat& edges, std::vector<Point2i>& contour) {
	std::vector<std::vector<Point2i>> contours;
	findAllContours(edges, contours);
	findLargestContour(contours, contour);
}

void detectLines(const Mat& edges, std::vector<Vec4i>& lines) {
	const double rho = 1.0;
	const double theta = M_PI / 180.0;
	const int threshold = 90;
	const int minLineLength = 100;
	const int maxLineGap = 80;

	HoughLinesP(edges, lines, rho, theta, threshold, minLineLength, maxLineGap);
}

double lineLength(cv::Vec4i line) {
	return sqrt((line[2] - line[0]) * (line[2] - line[0])
		+ (line[3] - line[1]) * (line[3] - line[1]));
}

std::vector<cv::Vec4i> removeSmallLines(const std::vector<cv::Vec4i>& lines, int minLength) {
	std::vector<Vec4i> result;

	std::copy_if(lines.begin(), lines.end(), std::back_inserter(result), [&](const auto& line) {
		return lineLength(line) >= minLength;
		});

	return result;
}

// source: https://stackoverflow.com/a/51121483
Vec2d linearParameters(Vec4i line) {
	Mat a = (Mat_<double>(2, 2) <<
		line[0], 1,
		line[2], 1);
	Mat y = (Mat_<double>(2, 1) <<
		line[1],
		line[3]);
	Vec2d mc; solve(a, y, mc);
	return mc;
}

// source: https://stackoverflow.com/a/51121483
Vec4i extendedLine(Vec4i line, double d) {
	// oriented left-t-right
	Vec4d _line = line[2] - line[0] < 0 ? Vec4d(line[2], line[3], line[0], line[1]) : Vec4d(line[0], line[1], line[2], line[3]);
	double m = linearParameters(_line)[0];
	// solution of pythagorean theorem and m = yd/xd
	double xd = sqrt(d * d / (m * m + 1));
	double yd = xd * m;
	return Vec4d(_line[0] - xd, _line[1] - yd, _line[2] + xd, _line[3] + yd);
}

// source: https://stackoverflow.com/a/51121483
std::vector<Point2i> boundingRectangleContour(Vec4i line, float d) {
	// finds coordinates of perpendicular lines with length d in both line points
	// https://math.stackexchange.com/a/2043065/183923

	Vec2f mc = linearParameters(line);
	float m = mc[0];
	float factor = sqrtf(
		(d * d) / (1 + (1 / (m * m)))
	);

	float x3, y3, x4, y4, x5, y5, x6, y6;
	// special case(vertical perpendicular line) when -1/m -> -infinity
	if (m == 0) {
		x3 = line[0]; y3 = line[1] + d;
		x4 = line[0]; y4 = line[1] - d;
		x5 = line[2]; y5 = line[3] + d;
		x6 = line[2]; y6 = line[3] - d;
	}
	else {
		// slope of perpendicular lines
		float m_per = -1 / m;

		// y1 = m_per * x1 + c_per
		float c_per1 = line[1] - m_per * line[0];
		float c_per2 = line[3] - m_per * line[2];

		// coordinates of perpendicular lines
		x3 = line[0] + factor; y3 = m_per * x3 + c_per1;
		x4 = line[0] - factor; y4 = m_per * x4 + c_per1;
		x5 = line[2] + factor; y5 = m_per * x5 + c_per2;
		x6 = line[2] - factor; y6 = m_per * x6 + c_per2;
	}

	return std::vector<Point2i> {
		Point2i(x3, y3),
			Point2i(x4, y4),
			Point2i(x6, y6),
			Point2i(x5, y5)
	};
}


// source: https://stackoverflow.com/a/51121483
bool extendedBoundingRectangleLineEquivalence(const Vec4i& _l1, const Vec4i& _l2, float extensionLengthFraction, float maxAngleDiff, float boundingRectangleThickness) {

	Vec4i l1(_l1), l2(_l2);
	// extend lines by percentage of line width
	float len1 = sqrtf((l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]));
	float len2 = sqrtf((l2[2] - l2[0]) * (l2[2] - l2[0]) + (l2[3] - l2[1]) * (l2[3] - l2[1]));
	Vec4i el1 = extendedLine(l1, len1 * extensionLengthFraction);
	Vec4i el2 = extendedLine(l2, len2 * extensionLengthFraction);

	// reject the lines that have wide difference in angles
	float a1 = atan(linearParameters(el1)[0]);
	float a2 = atan(linearParameters(el2)[0]);
	if (fabs(a1 - a2) > maxAngleDiff * M_PI / 180.0) {
		return false;
	}

	// calculate window around extended line
	// at least one point needs to inside extended bounding rectangle of other line,
	std::vector<Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
	return
		pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) == 1 ||
		pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) == 1;
}


// based on: https://stackoverflow.com/a/51121483
std::vector<cv::Vec4i> reduceLinesByEquivalency(const std::vector<cv::Vec4i>& lines) {
	// line extension length - as fraction of original line width
	const float extensionLengthFraction = 0.2;
	// maximum allowed angle difference for lines to be considered in same equivalence class
	const float maxAngleDiff = 2.0;
	// thickness of bounding rectangle around each line
	const float boundingRectangleThickness = 10;


	std::vector<int> labels; // unique for each equivalency class
	int equilavenceClassesCount = cv::partition(lines, labels, [&](const Vec4i l1, const Vec4i l2) {
		return extendedBoundingRectangleLineEquivalence(
			l1, l2, extensionLengthFraction, maxAngleDiff, boundingRectangleThickness);
		});

	// build point clouds out of each equivalence classes
	std::vector<std::vector<Point2i>> pointClouds(equilavenceClassesCount);
	for (int i = 0; i < lines.size(); i++) {
		const Vec4i& detectedLine = lines[i];
		pointClouds[labels[i]].push_back(Point2i(detectedLine[0], detectedLine[1]));
		pointClouds[labels[i]].push_back(Point2i(detectedLine[2], detectedLine[3]));
	}

	// fit line to each equivalence class point cloud
	std::vector<Vec4i> reducedLines = std::accumulate(pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& _pointCloud) {
		std::vector<Point2i> pointCloud = _pointCloud;

		//lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
		// (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
		Vec4f lineParams; fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

		// derive the bounding xs of point cloud
		decltype(pointCloud)::iterator minXP, maxXP;
		std::tie(minXP, maxXP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2) { return p1.x < p2.x; });

		// derive y coords of fitted line
		float m = lineParams[1] / lineParams[0];
		int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
		int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];

		target.push_back(Vec4i(minXP->x, y1, maxXP->x, y2));
		return target;
		});

	return reducedLines;
}

void reduceLines(std::vector<cv::Vec4i>& lines)
{
	const int min_line_length = 30;

	lines = removeSmallLines(lines, min_line_length);
	lines = reduceLinesByEquivalency(lines);
}

void discardExternalLines(const std::vector<Point2i>& contour, std::vector<cv::Vec4i>& lines, Size imageSize) {
	std::vector<Vec4i> clippedLines;
	
	Mat contourFrame = Mat::zeros(imageSize, CV_8UC1);

	drawContours(contourFrame, std::vector<std::vector<Point2i>>(1,  contour), -1, 255, 1, 8);
	
	for (const auto& line : lines) {
		Point2f a(line[0], line[1]);
		Point2f b(line[2], line[3]);

		Mat lineFrame = Mat::zeros(imageSize, CV_8UC1);
		cv::line(lineFrame, a, b, 255, 4);

		Mat result;
		bitwise_and(lineFrame, contourFrame, result);

		if(countNonZero(result) > 0)
			clippedLines.push_back(line);
	}
	
	lines = std::move(clippedLines);
}

template <typename _Tp>
bool isInside(const Size& imgSize, const _Tp& point) {
	return 0 <= point.y && point.y < imgSize.height && 0 <= point.x && point.x < imgSize.width;
}

std::pair<Point2i, Point2i> lineToPoints(const Vec4i line) {
	return {
		Point2i(line[0], line[1]),
		Point2i(line[2], line[3])
	};
}

bool intersectLines(Vec4i line1, Vec4i line2, Point2f& intersection)
{
	const auto [s1, e1] = lineToPoints(line1);
	const auto [s2, e2] = lineToPoints(line2);

	const Point dir1 = e1 - s1;
	const Point dir2 = e2 - s2;
	const Point dirStart = s2 - s1;

	const double cross = dir1.cross(dir2);

	if (cross == 0) // parallel lines
		return false;

	const double t1 = dirStart.cross(dir2) / cross;
	intersection = Point2f(s1) + Point2f(dir1) * t1;
	return true;
}

std::vector<cv::Point2f> findIntersections(const std::vector<cv::Vec4i>& lines, double minAngle) {
	std::vector<Point2f> intersections;

	for (int i = 0; i < lines.size(); i++) {
		for (int j = 0; j < lines.size(); j++) {
			if (i == j)
				continue;

			const Vec4i& line1 = lines[i];
			const Vec4i& line2 = lines[j];

			const auto [s1, e1] = lineToPoints(line1);
			const auto [s2, e2] = lineToPoints(line2);

			const Point dir1 = e1 - s1;
			const Point dir2 = e2 - s2;

			const double angle = acos(dir1.dot(dir2) / norm(dir1) / norm(dir2));

			if (angle < minAngle)
				continue;

			Point2f intersection;			

			if (intersectLines(line1, line2, intersection)) {
				intersections.push_back(intersection);
			}		
		}
	}

	return intersections;
}

std::vector<cv::Point2f> findIntersections(const std::vector<cv::Vec4i>& lines, cv::Size imageSize) {
	static const double minAngle = degToRad(20); // minimum angle between lines
	static const int thresholdPointDist = 20; //minimum distance between points

	std::vector<Point2f> intersections = findIntersections(lines, minAngle);
	clipPoints(intersections, imageSize);
	filterClosePoints(intersections, thresholdPointDist);
	return intersections;
}


template<typename _Tp>
void clipPoints(std::vector<_Tp>& points, cv::Size clipWindow) {
	points.erase(
		std::remove_if(
			points.begin(),
			points.end(),
			[&](const auto& point) {
				return !isInside(clipWindow, point);
			}),
		points.end());
}

template<typename _Tp>
bool isCloseToOtherPoints(const _Tp& point, const std::vector<_Tp>& points, double thresholdDistance) {
	for (const auto& point2 : points) {
		if (norm(point - point2) < thresholdDistance) {
			return true;
		}
	}
	return false;
}

template<typename _Tp>
void filterClosePoints(std::vector<_Tp>& points, double thresholdDistance) {
	std::vector<_Tp> savedPoints;

	for (const auto& point : points) {
		if (!isCloseToOtherPoints(point, savedPoints, thresholdDistance)) {
			savedPoints.push_back(point);
		}
	}

	points = std::move(savedPoints);
}

template<typename _Tp>
void reduceConvexHull(std::vector<_Tp>& convexHull, size_t max_size) {
	while (convexHull.size() > max_size) {
		const size_t size = convexHull.size();
		size_t remove_i = -1;
		double min_dist = 0;
		
		for (size_t curr_i = 0; curr_i < size; curr_i++) {

			size_t prev_i = (curr_i - 1 + size) % size;
			size_t next_i = (curr_i + 1) % size;

			_Tp a = convexHull[prev_i];
			_Tp b = convexHull[next_i];
			_Tp p = convexHull[curr_i];

			double dist_p_ab = norm((b - a).cross(a - p)) / norm(b - a);

			if (curr_i == 0 || dist_p_ab < min_dist) {
				min_dist = dist_p_ab;
				remove_i = curr_i;
			}
		}
		convexHull.erase(convexHull.begin() + remove_i);
	}
}


// explicit instantiations
template bool isInside<Point2i>(const Size& imgSize, const Point2i& point);
template bool isInside<Point2f>(const Size& imgSize, const Point2f& point);
template void clipPoints<Point2i>(std::vector<Point2i>& points, cv::Size clipWindow);
template void clipPoints<Point2f>(std::vector<Point2f>& points, cv::Size clipWindow);
template bool isCloseToOtherPoints<Point2i>(const Point2i& point, const std::vector<Point2i>& points, double thresholdDistance);
template bool isCloseToOtherPoints<Point2f>(const Point2f& point, const std::vector<Point2f>& points, double thresholdDistance);
template void filterClosePoints<Point2i>(std::vector<Point2i>& points, double thresholdDistance);
template void filterClosePoints<Point2f>(std::vector<Point2f>& points, double thresholdDistance);
template void reduceConvexHull<Point2i>(std::vector<Point2i>& convexHull, size_t max_size);
template void reduceConvexHull<Point2f>(std::vector<Point2f>& convexHull, size_t max_size);
