// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <random>
#include <algorithm>
#include <numeric>

#include "common.h"
#include "Menu.h"
#include "Menu.h"
#include "config.h"
#include "util.h"
#include "colors.h"
#include "calibration.h"
#include "chessboard.h"
#include "visualization.h"
#include "windows_fs.h"

using namespace cv;

bool comparePoints(const Point2f& p1, const Point2f& p2) {
	return std::tie(p1.x, p1.y) < std::tie(p2.x, p2.y);
}


void waitForKey() {
	std::cout << "Press any key...." << std::endl;
	std::cin.get();
}

void show_image(std::string name, const Mat& img) {
	imshow(name, img);
	imwrite(name + ".png", img);
}

template<typename _Tp>
void show_contour(std::string name, const Mat& background_img, const std::vector<_Tp>& contour) {
	const Vec3b color_contour(0, 0, 255);

	std::vector<std::vector<_Tp>> contours(1, contour);
	Mat imgWithContour = background_img.clone();

	drawContours(imgWithContour, contours, -1, color_contour, 1, 8);
	show_image(name, imgWithContour);
}

void show_lines(std::string name, const Mat& background_img, const std::vector<Vec4i>& lines) {
	Mat img = background_img.clone();

	for (const Vec4i& line : lines) {
		Scalar color;

		if (img.channels() == 1) {
			color = genRandomGrayColor();
		}
		else {
			color = genUniqueBGRColor();
		}

		cv::line(img,
			cv::Point(line[0], line[1]),
			cv::Point(line[2], line[3]), color, 2);
	}

	show_image(name, img);
}

template<typename _Tp>
void draw_points(Mat& img, const std::vector<_Tp>& points) {

	for (const auto& point : points) {
		circle(img, point, 5, Vec3b(0, 0, 255), -1);
	}
}

template<typename _Tp>
void show_points (std::string name, const Mat& background_img, const std::vector<_Tp>& points) {
	Mat img = background_img.clone();

	draw_points(img, points);

	show_image(name, img);
}

template<typename _Tp>
void print_points(std::string message, const std::vector<_Tp>& points) {
	std::cout << message << ": ";
	for (const auto& point : points) {
		std::cout << point << " ";
	}
	std::cout << "\n";
}

std::vector<Point2f> rectangleToPoints(cv::Rect2f rect) {
	return {
		rect.tl(),
		rect.tl() + Point2f(0, rect.height),
		rect.br(),
		rect.tl() + Point2f(rect.width, 0)
	};
}

void show_boundingBox(std::string name, const Mat& background_img, const std::vector<Point2f>& boundingBoxPoints) {
	Mat img = background_img.clone();

	draw_points(img, boundingBoxPoints);
	for (int i = 0; i < boundingBoxPoints.size(); i++) {
		int j = (i + 1) % boundingBoxPoints.size();
		line(img, boundingBoxPoints[i], boundingBoxPoints[j], Vec3b(0, 255, 0));
	}

	show_image(name, img);
}

void show_boundingBox (std::string name, const Mat& background_img, Rect2f boundingBox) {
	show_boundingBox(name, background_img, rectangleToPoints(boundingBox));
}

void testCameraCalibration(Mat cameraMatrix, const Mat& distCoeffs)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_COLOR);
		Size imgSize = img.size();

		std::vector<Point2f> corners;

		bool found = detectCorners(img, boardSize, winSize, corners);

		drawChessboardCorners(img, boardSize, Mat(corners), found);


	/*	float scaleX = (float) img.cols / (float) calibImgSize.width;
		float scaleY = (float) img.rows / (float) calibImgSize.height;

		std::cout << cameraMatrix.at<double>(0, 0) << "\n";
		cameraMatrix.at<double>(0, 0) *= scaleX;
		std::cout << cameraMatrix.at<double>(0, 0) << "\n";
		cameraMatrix.at<double>(1, 1) *= scaleY;
		cameraMatrix.at<double>(0, 2) *= scaleX;
		cameraMatrix.at<double>(1, 2) *= scaleY;

		std::cout << "cameraMatrix:\n" << cameraMatrix << "\n";*/


		Mat optCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0);

		Mat dst = img.clone();
		//undistort(img, dst, cameraMatrix, distCoeffs);

		Mat map1, map2;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), optCameraMatrix, imgSize,CV_16SC2, map1, map2);
		remap(img, dst, map1, map2, INTER_LINEAR);

		imshow("source image", img);
		imshow("undistorted image", dst);
		imwrite("result.png", dst);

		waitKey();
	}
}

void testCameraCalibration()
{
	//cameraCalibration();

	const std::vector<std::string> imgPaths = getFilesInDir(calibrationImageDir);
	std::vector<std::vector<Point2f>> imgCorners;
	std::vector<Point3f> objCorners;

	Mat cameraMatrix;
	Mat distCoeffs;
	std::vector<Mat> rvecs;
	std::vector<Mat> tvecs;

	Mat img = imread(imgPaths[0], IMREAD_COLOR);
	Size imgSize = img.size();
	//Size imgSize = calibImgSize;
	//Size imgSize = Size(10, 10);



	detectCorners(imgPaths, boardSize, winSize, imgCorners);

	if (imgCorners.empty()) {
		std::cerr << "No corner was found in the calibrating images\n";
		return;
	}

	calcBoardCornerPositions(boardSize, squareSize, objCorners);
	bool result = calibrateCamera(imgCorners, imgSize, objCorners, cameraMatrix, distCoeffs, rvecs, tvecs);

	if (!result) {
		std::cerr << "Calibration failed\n";
		return;
	}

	std::cout << "cameraMatrix:\n" << cameraMatrix << "\n";
	/*std::cout << "distCoeffs:\n" << distCoeffs << "\n";
	std::cout << "rotVec:\n";
	for (const auto& rvec : rvecs) {
		std::cout << rvec << "\n";
	}
	std::cout << "\n";
	std::cout << "transVec:\n";
	for (const auto& tvec : tvecs) {
		std::cout << tvec << "\n";
	}
	std::cout << "\n";*/


	testCameraCalibration(cameraMatrix, distCoeffs);


	//waitForKey();
	//waitKey();


	/*assert(imgPaths.size() == imgPoints.size());

	int i = 0;
	for (const auto& path : imgPaths) {
		Mat img = imread(path, IMREAD_COLOR);

		drawChessboardCorners(img, boardSize, Mat(imgPoints[i++]), true);

		imshow("image", img);

		waitKey();
	}*/
}

void testCornerDetection()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_COLOR);

		std::vector<Point2f> corners;

		bool found = detectCorners(img, boardSize, winSize, corners);

		drawChessboardCorners(img, boardSize, Mat(corners), found);

		imshow("image", img);
		imwrite("result.png", img);

		waitKey();
	}
}

void printCalibrationImagePaths() {
	std::vector<std::string> imagePaths = getFilesInDir(calibrationImageDir);

	std::cout << "Calibration image directory: " << calibrationImageDir << "\n";
	std::cout << "Calibration images: \n";

	for (const auto& path : imagePaths) {
		std::cout << "\t" << path << "\n";
	}

	waitForKey();
}

//void adjustBrightness(Mat& img) {
//	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
//	Mat closed;
//	morphologyEx(img, closed, MORPH_CLOSE, kernel);
//	Mat floatMat;
//	img.convertTo(floatMat, CV_32FC1);
//	floatMat /= closed;
//	normalize(floatMat, floatMat, 0, 255, NORM_MINMAX);
//	floatMat.convertTo(img, img.type());
//}
//

//
//
//
//bool isEqual(const Vec4i& _l1, const Vec4i& _l2)
//{
//	Vec4i l1(_l1), l2(_l2);
//
//	float length1 = sqrtf((l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]));
//	float length2 = sqrtf((l2[2] - l2[0]) * (l2[2] - l2[0]) + (l2[3] - l2[1]) * (l2[3] - l2[1]));
//
//	float product = (l1[2] - l1[0]) * (l2[2] - l2[0]) + (l1[3] - l1[1]) * (l2[3] - l2[1]);
//
//	if (fabs(product / (length1 * length2)) < cos(CV_PI / 30))
//		return false;
//
//	float mx1 = (l1[0] + l1[2]) * 0.5f;
//	float mx2 = (l2[0] + l2[2]) * 0.5f;
//
//	float my1 = (l1[1] + l1[3]) * 0.5f;
//	float my2 = (l2[1] + l2[3]) * 0.5f;
//	float dist = sqrtf((mx1 - mx2) * (mx1 - mx2) + (my1 - my2) * (my1 - my2));
//
//	if (dist > max(length1, length2) * 0.5f)
//		return false;
//
//	return true;
//}
//


//
//// 4 points of a retangle
//void getFourCorners(const std::vector<Point2f>& corners, Point2i& topLeft, Point2i& bottomLeft, Point2i& topRight, Point2i& bottomRight) {
//	topLeft = bottomLeft = topRight = bottomRight = corners[0];
//
//	for (int i = 1; i < corners.size(); i++) {
//		const Point2i& p = corners[i];
//
//		if (p.x <= topLeft.x && p.y <= topLeft.y) {
//			topLeft = p;
//		}
//
//		if (p.x <= bottomLeft.x && p.y >= bottomLeft.y) {
//			bottomLeft = p;
//		}
//
//		if (p.x >= topRight.x && p.y <= topRight.y) {
//			topRight = p;
//		}
//
//		if (p.x >= bottomRight.x && p.y >= bottomRight.y) {
//			bottomRight = p;
//		}
//	}
//}
//
//bool isLineInsidePolygon(const std::vector<Point>& polygon, Vec4i line, Size imageSize) {
//	Point2f a = Point2f(line[0], line[1]);
//	Point2f b = Point2f(line[2], line[3]);
//	/*Point2f vec = b - a;
//	double l = norm(vec);
//	vec /= l;
//
//
//	Point2f t = -a;
//	double angle = -acos(vec.dot(Point2f(1.0f, 0.0f)));
//	double cos_a = cos(angle);
//	double sin_a = sin(angle);
//
//	int nrint = 0;
//
//	for (int i = 0; i < polygon.size(); i++) {
//		int next = (i + 1) % polygon.size();
//
//		Point2f p1 = polygon[i];
//		p1 += t;
//		p1 = Point2f(
//			cos_a * p1.x - sin_a * p1.y,
//			sin_a * p1.x + cos_a * p1.y
//		);
//
//
//		Point2f p2 = polygon[next];
//		p2 += t;
//		p2 = Point2f(
//			cos_a * p2.x - sin_a * p2.y,
//			sin_a * p2.x + cos_a * p2.y
//		);
//
//		if (p1.y == 0 || (p1.y > 0) == (p2.y > 0)) {
//			continue;
//		}
//
//		double intX = (0.0f - p1.y) / (p2.y - p1.y) * (p2.x - p1.x);
//
//		if (0 < intX && intX < l)
//			return false;
//
//		if(intX < 0)
//			nrint++;
//	}
//
//
//	return nrint % 2 == 1;*/
//
//	//return pointPolygonTest(polygon, a, false) >= 0 || pointPolygonTest(polygon, b, false) >= 0;
//
//	Mat lineFrame = Mat(imageSize, CV_8UC1);
//	cv::line(lineFrame, a, b, 255, 4);
//
//	//Mat contourFrame = Mat(imageSize, CV_8UC1);
//	//drawContours(contourFrame, polygon, )
//	return false;
//}
//
//int hough_threshold = 90;
//
//static void CannyThreshold2(int, void*)
//{

//	//std::vector<Point2i> contour_hull;
//	//convexHull(contours[0], contour_hull);
//	//contours = std::vector<std::vector<Point2i>>(1, contour_hull);
//
//	//imgWithContours = src.clone();
//	//drawContours(imgWithContours, contours, -1, Vec3b(0, 0, 255), 1, 8);
//	//rectangle(imgWithContours, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
//	//imshow("Contour convex hull", imgWithContours);
//
//
//	int bias = 7;
//	bounding_rect.x = max(0, bounding_rect.x - bias);
//	bounding_rect.y = max(0, bounding_rect.y - bias);
//	bounding_rect.width = min(bounding_rect.width + bias, src.cols - 1 - bounding_rect.x);
//	bounding_rect.height = min(bounding_rect.height + bias, src.cols - 1 - bounding_rect.y);
//
//	//src = src(bounding_rect);
//	//src_gray = src_gray(bounding_rect);
//
//
//


//
//	Mat reprojected;
//	std::vector<Point2f> reprojectedCorners;
//	reprojectImage(src, reprojected, convex_hull, imageSize);
//
//	
//	Mat reprojectedWith4Corners = reprojected.clone();
//	reprojectPoints(corners, reprojectedCorners, corners, imageSize);
//	for (const auto& corner : reprojectedCorners) {
//		circle(reprojectedWith4Corners, corner, 5, Vec3b(0, 0, 255), -1);
//	}
//
//	imshow("Reprojected image with four corners", reprojectedWith4Corners);
//	imwrite("Reprojected image with four corners.png", reprojectedWith4Corners);
//
//
//
//	Point2i topLeft, bottomLeft, topRight, bottomRight;
//	getFourCorners(reprojectedCorners, topLeft, bottomLeft, topRight, bottomRight);
//
//	float dx = (topRight.x - topLeft.x) / 8.0f;
//	float dy = (bottomLeft.y - topLeft.y) / 8.0f;
//
//
//	std::vector<Point2f> allReprCorners;
//	for (int i = 0; i <= 8; i++) {
//		for (int j = 0; j <= 8; j++) {
//			allReprCorners.push_back(topLeft + Point2i(j * dx, i * dy));
//		}
//	}
//
//	//std::vector<Point2f> allReprCorners;
//	//std::vector<Point2f> intersectionsf;
//	//for (const auto& inter : intersections) {
//	//	intersectionsf.push_back(inter);
//	//}
//
//	//reprojectPoints(intersectionsf, allReprCorners, corners, imageSize);
//
//
//	Mat reprojectedWithAllCorners = reprojected.clone();
//	for (const auto& corner : allReprCorners) {
//		circle(reprojectedWithAllCorners, corner, 5, Vec3b(0, 0, 255), -1);
//	}
//	imshow("Reprojected image with all corners", reprojectedWithAllCorners);
//	imwrite("Reprojected image with all corners.png", reprojectedWithAllCorners);
//
//}

struct Margin {
	float top = 0.0f;
	float bottom = 0.0f;
	float left = 0.0f;
	float right = 0.0f;
};

// assumes counterclockwise order, starting from top-left
std::vector<Point2f> createCorners(cv::Size imageSize, Margin margin = {}) {
	return {
	{0 + margin.left, 0 + margin.top}, // top-left
	{0 + margin.left, imageSize.height - margin.bottom}, // bottom-left
	{imageSize.width - margin.right, imageSize.height - margin.bottom}, // bottom-right
	{imageSize.width - margin.right, 0 + margin.top}, // top-right
	};
}

// assumes counterclockwise order, starting from top-left
void calcProjectionParams(const std::vector<Point2f>& corners, Mat& homography, Mat& perspective, cv::Size imageSize, Margin margin = {}) {
	std::vector<Point2f> repr_corners = createCorners(imageSize, margin);
	homography = findHomography(corners, repr_corners);
	perspective = getPerspectiveTransform(corners, repr_corners);
}


inline double getClockwiseAngle(Point2f p) {
	return -atan2(p.x, -p.y);;
}

Point2f findCentroid(const std::vector<Point2f>& points) {
	Point2f center;

	for (const auto& p : points) {
		center += p;
	}

	return center / (float)points.size();
} 

void sortPoints(std::vector<Point2f>& points, bool clockWise) {
	Point2f center = findCentroid(points);

	std::sort(points.begin(), points.end(), [&](const auto& p1, const auto& p2) {
		if (clockWise)
			return getClockwiseAngle(p1 - center) > getClockwiseAngle(p2 - center);
		else
			return getClockwiseAngle(p1 - center) < getClockwiseAngle(p2 - center);
		});
}

void perspectiveTransformRectangle(cv::Rect2f rect, std::vector<Point2f>& points, cv::InputArray perspective) {
	perspectiveTransform(rectangleToPoints(rect), points, perspective);
}

template <typename _Tp>
std::vector<Point2f> toPoint2fVec(std::vector<_Tp> src) {
	std::vector<Point2f> dst(src.size());
	std::transform(src.begin(), src.end(), dst.begin(), [](const auto& p) { return p; });
	return dst;
}


Margin calcOptimalMargin(const std::vector<Point2f>& corners, const std::vector<Point2f>& contour, cv::Size imageSize) {
	const float bias = 10;
	std::vector<Point2f> contour_r;
	
	Mat perspective = getPerspectiveTransform(corners, createCorners(imageSize));
	perspectiveTransform(contour, contour_r, perspective);

	Rect2f contourBB = boundingRect(contour_r);
	std::vector<Point2f> contourBBPoints = rectangleToPoints(contourBB);

	float minX = contourBBPoints[0].x;
	float maxX = contourBBPoints[0].x;
	float minY = contourBBPoints[0].y;
	float maxY = contourBBPoints[0].y;

	for (const auto& point : contourBBPoints) {
		minX = min(minX, contourBBPoints[0].x);
		maxX = max(maxX, contourBBPoints[0].x);
		minY = min(minY, contourBBPoints[0].y);
		maxY = max(maxY, contourBBPoints[0].y);
	}

	return {
		abs(std::min(0.0f, minY)) + bias,
		std::max(0.0f, maxY - imageSize.height) + bias,
		abs(std::min(0.0f, minX)) + bias,
		std::max(0.0f, maxX - imageSize.width) + bias
	};
}

std::vector<Point2f> computeLatticePoints(Point2f topLeft, Point2f bottomLeft, Point2f bottomRight, Point2f topRight) {
	std::vector<Point2f> lattice;
	lattice.reserve(9 * 9);
	
	const float dx = (topRight.x - topLeft.x) / 8.0f;
	const float dy = (bottomLeft.y - topLeft.y) / 8.0f;

	for (int i = 0; i <= 8; i++) {
		for (int j = 0; j <= 8; j++) {
			lattice.push_back(topLeft + Point2f(j * dx, i * dy));
		}
	}

	return lattice;
}

void testChessboardDetection() {

	/* Images */
	Mat src; // source image
	Mat src_gray; // source image converted to grayscale
	Mat detected_edges; // binary image containing the edges of the image
	Mat img_r; // reprojection of the source image
	Mat dst;

	/* Matrices */
	Mat homography; // image wrapping
	Mat perspective; // perspective matrix - point transformation

	/* Variables */
	std::vector<Point2i> contour; // contour of the chessboard grid
	std::vector<Point2f> contour_r; // reprojected contour of the chessboard grid
	std::vector<Vec4i> lines; // lines of the chessboard
	std::vector<Point2f> intersections; // intersections of the chessboard lines
	std::vector<Point2f> corners; // corners of the chessboard grid
	std::vector<Point2f> lattice; // corners of the chessboard grid

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		// 1. Read image, resize
		src = imread(fname, IMREAD_COLOR);
		resize(src, src, imageSize);
		show_image("source image", src);

		// 2. Convert source image to grayscale
		cvtColor(src, src_gray, COLOR_BGR2GRAY);
		show_image("source image - gray", src_gray);

		// 3. Detect edges
		detectEdges(src_gray, detected_edges);
		show_image("detected edges", detected_edges);

		// 4. Contour detection
		findLargestContour(detected_edges, contour);
		show_contour("contour", src, contour);

		// 5. Line detection
		detectLines(detected_edges, lines);
		std::cout << "Number of detected lines: " << lines.size() << std::endl;
		show_lines("detected lines", src, lines);
		show_lines("detected lines - lines only", Mat::zeros(src.size(), src.type()), lines);

		// 5.1 Reduce the number of lines
		reduceLines(lines);
		std::cout << "Reduced number of lines: " << lines.size() << std::endl;
		show_lines("reduced lines", src, lines);
		show_lines("reduced lines - lines only", Mat::zeros(src.size(), src.type()), lines);

		// 5.2 Discard the lines that are entirely outside of the contour
		discardExternalLines(contour, lines, src.size());
		std::cout << "Number of filtered lines: " << lines.size() << std::endl;
		show_lines("filtered lines", src, lines);
		show_lines("filtered lines - lines only", Mat::zeros(src.size(), src.type()), lines);


		// 6. Finding the intersection points of the chessboard lines
		intersections = findIntersections(lines, src.size());
		std::cout << "Number of intersections: " << intersections.size() << std::endl;
		show_points("intersection points", src, intersections);

		// 7. finding the four corners of the chessboard grid
		// 7.1 Finding the convex hull
		convexHull(intersections, corners);
		std::cout << "Number of elements in the convex hull: " << corners.size() << std::endl;
		show_points("convex hull", src, corners);

		if (corners.size() < 4) {
			std::cerr << "the 4 corners were not found";
			exit(1);
		}

		// 7.2 Reduce the convex hull to the four corners
		reduceConvexHull(corners, 4);
		assert(corners.size() == 4);

		// 7.3 Sort the corners in counterclockwise order. 
		sortPoints(corners, false);  //The bottom - right corner is at the front.

		// 7.3.1 Rotate the corners in order to put the top-right corner to the front
		// this way the board will be oriented such that the H1 cell will be in the bottom-right.
		std::rotate(corners.begin(), corners.begin() + 1, corners.end());
		show_points("Corners", src, corners);
		print_points("Corners", corners);


		// 8. Reprojection
		// 8.1 Compute the optimal margin
		// computes a margin such that all pieces are completely inside the image
		// purpose:
		//	- visualization
		//	- needed if the pieces are cropped separately (not used)
		Margin margin = calcOptimalMargin(corners, toPoint2fVec(contour), src.size());

		// 8.2 Compute transformation matrices
		calcProjectionParams(corners, homography, perspective, src.size(), margin);

		// 8.3. Reproject image
		warpPerspective(src, img_r, homography, src.size());
		show_image("reprojected", img_r);

		// 8.4 Reproject corner points
		perspectiveTransform(corners, corners, perspective);
		show_points("Reprojected image with four corners", img_r, corners);

		// 8.5 Reproject contour points
		perspectiveTransform(toPoint2fVec(contour), contour_r, perspective);
		// display bounding rectangle of the contour
		show_boundingBox("Reprojected image with contour bounding box", img_r, boundingRect(contour_r));


		// 9. Compute lattice points
		lattice = computeLatticePoints(corners[0], corners[1], corners[2], corners[3]);
		show_points("Reprojected image with all lattice points", img_r, lattice);

		waitKey();
	}
}


void testVisualizeChessboard() {
	std::vector<std::pair<Piece, Point2i>> pieces{
		{Piece::WhiteRook, C_A1},
		{Piece::WhiteKnight, C_B1},
		{Piece::WhiteBishop, C_C1},
		{Piece::WhiteQueen, C_D1},
		{Piece::WhiteKing, C_E1},
		{Piece::WhiteBishop, C_F1},
		{Piece::WhiteKnight, C_G1},
		{Piece::WhiteRook, C_H1},
		{Piece::WhitePawn, C_A2},
		{Piece::WhitePawn, C_B2},
		{Piece::WhitePawn, C_C2},
		{Piece::WhitePawn, C_D2},
		{Piece::WhitePawn, C_E4},
		{Piece::WhitePawn, C_F2},
		{Piece::WhitePawn, C_G2},
		{Piece::WhitePawn, C_H2},
		{Piece::BlackRook, C_A8},
		{Piece::BlackKnight, C_B8},
		{Piece::BlackBishop, C_C8},
		{Piece::BlackQueen, C_D8},
		{Piece::BlackKing, C_E8},
		{Piece::BlackBishop, C_F8},
		{Piece::BlackKnight, C_G8},
		{Piece::BlackRook, C_H8},
		{Piece::BlackPawn, C_A7},
		{Piece::BlackPawn, C_B7},
		{Piece::BlackPawn, C_C7},
		{Piece::BlackPawn, C_D7},
		{Piece::BlackPawn, C_E5},
		{Piece::BlackPawn, C_F7},
		{Piece::BlackPawn, C_G7},
		{Piece::BlackPawn, C_H7},
	};

	Mat dst = getDigitalChessboard(pieces);
	show_image("Chessboard", dst);

	waitKey(0);
}

int main()
{
	Menu menu({
		{"Corner detection", testCornerDetection},
		{"Print the paths of the calibration images", printCalibrationImagePaths},
		{"Camera calibration", testCameraCalibration},
		{"Chessboard detection", testChessboardDetection},
		{"Chessboard visualization", testVisualizeChessboard}
		});

	menu.show(std::cin, std::cout);
	return 0;
}