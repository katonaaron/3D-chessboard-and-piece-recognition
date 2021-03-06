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
#include "tensorflow.h"

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

void draw_boundingBox(Mat& img, const std::vector<Point2f>& boundingBoxPoints, cv::Scalar color = Vec3b(0, 255, 0)) {
	for (size_t i = 0; i < boundingBoxPoints.size(); i++) {
		size_t j = (i + 1) % boundingBoxPoints.size();
		line(img, boundingBoxPoints[i], boundingBoxPoints[j], color);
	}
}

void show_boundingBox(std::string name, const Mat& background_img, const std::vector<Point2f>& boundingBoxPoints, bool drawPoints = true, cv::Scalar color = Vec3b(0, 255, 0)) {
	Mat img = background_img.clone();

	if(drawPoints)
		draw_points(img, boundingBoxPoints);
	draw_boundingBox(img, boundingBoxPoints);
	show_image(name, img);
}

void show_boundingBoxes(std::string name, const Mat& background_img, const std::vector<std::vector<Point2f>>& boundingBoxes, bool drawPoints = true, cv::Scalar color = Vec3b(0, 255, 0)) {
	Mat img = background_img.clone();

	for (const auto& bbox : boundingBoxes) {
		if (drawPoints)
			draw_points(img, bbox);
		draw_boundingBox(img, bbox);
	}

	show_image(name, img);
}

void show_boundingBox (std::string name, const Mat& background_img, Rect2f boundingBox) {
	show_boundingBox(name, background_img, rectangleToPoints(boundingBox));
}

Piece classToPiece(int label) {
	switch (label)
	{
	case 0:
		std::cerr << "Unlabeled item\n";
		exit(1);
	case 1:
	case 8:
		return Piece::WhiteBishop;
	case 2:
		return Piece::BlackBishop;
	case 3:
		return Piece::BlackKing;
	case 4:
		return Piece::BlackKnight;
	case 5:
		return Piece::BlackPawn;
	case 6:
		return Piece::BlackQueen;
	case 7:
		return Piece::BlackRook;
	case 9:
		return Piece::WhiteKing;
	case 10:
		return Piece::WhiteKnight;
	case 11:
		return Piece::WhitePawn;
	case 12:
		return Piece::WhiteQueen;
	case 13:
		return Piece::WhiteRook;
	default:
		std::cerr << "Invalid label\n";
		exit(1);
		break;
	}
}

void print_predictions(const Prediction& pred, float thresholdScore = 0.0f) {
	for (int i = 0; i < pred.num_detections; i++) {
		if (pred.scores[i] < thresholdScore)
			continue;
		std::cout << "#" << i << ":\n";
		std::cout << "\t class: " << pieceToString(classToPiece(pred.classes[i])) << "\n";
		std::cout << "\t score: " << pred.scores[i] << "\n";
		std::cout << "\t box: ";
		for (const auto& cord : pred.boxes[i]) {
			std::cout << cord << " ";
		}
		std::cout << "\n";
	}

}

void print_prediction_statistics(const Prediction& pred) {
	std::cout << "num_detections: " << pred.num_detections << "\n";
	std::cout << "score statistics:\n";
	std::cout << "\tmin: " << *std::min_element(pred.scores.begin(), pred.scores.end()) << "\n";
	std::cout << "\tmean: " << accumulate(pred.scores.begin(), pred.scores.end(), 0.0) / pred.scores.size() << "\n";
	std::cout << "\tmax: " << *std::max_element(pred.scores.begin(), pred.scores.end()) << "\n";
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

cv::Rect boundingBoxToRect(const std::vector<float>& box, cv::Size imageSize) {
	int ymin = (int)(box[0] * imageSize.height);
	int xmin = (int)(box[1] * imageSize.width);
	int h = (int)(box[2] * imageSize.height) - ymin;
	int w = (int)(box[3] * imageSize.width) - xmin;
	return Rect(xmin, ymin, w, h);
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


void testPieceRecognition() {
	//saving current path
	auto path = std::filesystem::current_path();
	const float threshold_score = 0.5f;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_COLOR);
		resize(img, img, config.imageSize);
		Prediction pred;

		//setting path back
		std::filesystem::current_path(path); 

		Model model(config.path_model_graph);
		std::cout << "Prediction started\n";
		model.predict(img, pred);

		if (pred.num_detections == 0) {
			std::cerr << "No piece was found\n";
			exit(1);
		}

		std::cout << "num_detections: " << pred.num_detections << "\n";
		std::cout << "score statistics:\n";
		std::cout << "\tmin: " << *std::min_element(pred.scores.begin(), pred.scores.end()) << "\n";
		std::cout << "\tmean: " << accumulate(pred.scores.begin(), pred.scores.end(), 0.0) / pred.scores.size() << "\n";
		std::cout << "\tmax: " << *std::max_element(pred.scores.begin(), pred.scores.end()) << "\n";

		print_predictions(pred, threshold_score);

		Size size = img.size();
		int height = size.height;
		int width = size.width;

		for (int i = 0; i < pred.num_detections; i++) {
			auto box = pred.boxes[i];
			auto score = pred.scores[i];
			if (score < threshold_score) {
				continue;
			}
			int ymin = (int)(box[0] * height);
			int xmin = (int)(box[1] * width);
			int h = (int)(box[2] * height) - ymin;
			int w = (int)(box[3] * width) - xmin;
			Rect rect = Rect(xmin, ymin, w, h);
			rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
		}

		show_image("piece recognition", img);

		waitKey();
	}
}

int findClosest(float x, float width) {
	float dx = width / 8;

	float q = x / dx;
	int closest = static_cast<int>(std::roundf(q)) - 1;

	if (closest < 0)
		closest = 0;
	if (closest > 7)
		closest = 7;
	return closest;
}

std::vector<std::pair<Piece, Point2i>> recreateChessboard(const std::vector<Piece>& pieces, const std::vector<float>& piece_scores, const std::vector<std::vector<Point2f>>& piece_boxes, const std::vector<Point2f>& corners) {
	Piece board[8][8];
	float scores[8][8];
	bool isOccupied[8][8] = {0};

	Point2f topLeft = corners[0];

	float width = corners[3].x - corners[0].x;
	float height = corners[1].y - corners[0].y;

	for (int i = 0; i < pieces.size(); i++) {
		const Piece& piece = pieces[i];
		const float& score = piece_scores[i];
		const std::vector<Point2f>& box = piece_boxes[i];

		float maxX = box[0].x;
		float maxY = box[0].y;

		for (const auto& point : box) {
			maxX = max(maxX, point.x);
			maxY = max(maxY, point.y);
		}

		float x = maxX - topLeft.x;
		float y = maxY - topLeft.y;

		Point pos = {
			findClosest(x, width),
			findClosest(y, height),
		};

		assert(0 <= pos.x && pos.x < 8);
		assert(0 <= pos.y && pos.y < 8);

		if (isOccupied[pos.x][pos.y]) {
			if (scores[pos.x][pos.y] > score) {
				continue;
			}
		}

		isOccupied[pos.x][pos.y] = true;
		board[pos.x][pos.y] = piece;
		scores[pos.x][pos.y] = score;
	}

	std::vector<std::pair<Piece, Point2i>> result;

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			if (isOccupied[i][j]) {
				result.emplace_back(board[i][j], Point2i(i, j));
			}
		}
	}

	return result;
}

void testChessboardDetectionAndPieceRecognition() {

	/* Images */
	Mat img; // source image
	Mat img_gray; // source image converted to grayscale
	Mat detected_edges; // binary image containing the edges of the image
	Mat img_r; // reprojection of the source image

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

	Model model(config.path_model_graph); // piece classifier model
	Prediction pred; // model prediction
	int nr_pieces = 0;
	std::vector<Piece> piece_types; // piece types
	std::vector<float> piece_scores; // piece scores;
	std::vector<Rect2f> piece_boxes; // piece boundig box points
	std::vector<std::vector<Point2f>> piece_boxes_r; // reprojected bounding box points, obtained by the prediction
	std::vector<std::pair<Piece, Point2i>> chessboard; // pieces and their positions, used for visualization

	/* Constants */
	const float threshold_score = 0.5f;
	const Size imageSize = config.imageSize;

	auto path = std::filesystem::current_path();
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		//setting path back
		std::filesystem::current_path(path);

		// 1. Read image, resize
		img = imread(fname, IMREAD_COLOR);
		resize(img, img, imageSize);
		show_image("source image", img);

		// 2. Convert source image to grayscale
		cvtColor(img, img_gray, COLOR_BGR2GRAY);
		show_image("source image - gray", img_gray);

		// 3. Detect edges
		detectEdges(img_gray, detected_edges);
		show_image("detected edges", detected_edges);

		// 4. Contour detection
		findLargestContour(detected_edges, contour);
		show_contour("contour", img, contour);

		// 5. Line detection
		detectLines(detected_edges, lines);
		std::cout << "Number of detected lines: " << lines.size() << std::endl;
		show_lines("detected lines", img, lines);
		show_lines("detected lines - lines only", Mat::zeros(img.size(), img.type()), lines);

		// 5.1 Reduce the number of lines
		reduceLines(lines);
		std::cout << "Reduced number of lines: " << lines.size() << std::endl;
		show_lines("reduced lines", img, lines);
		show_lines("reduced lines - lines only", Mat::zeros(img.size(), img.type()), lines);

		// 5.2 Discard the lines that are entirely outside of the contour
		discardExternalLines(contour, lines, img.size());
		std::cout << "Number of filtered lines: " << lines.size() << std::endl;
		show_lines("filtered lines", img, lines);
		show_lines("filtered lines - lines only", Mat::zeros(img.size(), img.type()), lines);


		// 6. Finding the intersection points of the chessboard lines
		intersections = findIntersections(lines, img.size());
		std::cout << "Number of intersections: " << intersections.size() << std::endl;
		show_points("intersection points", img, intersections);

		// 7. finding the four corners of the chessboard grid
		// 7.1 Finding the convex hull
		convexHull(intersections, corners);
		std::cout << "Number of elements in the convex hull: " << corners.size() << std::endl;
		show_points("convex hull", img, corners);

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
		show_points("Corners", img, corners);
		print_points("Corners", corners);


		// 8. Reprojection
		// 8.1 Compute the optimal margin
		// computes a margin such that all pieces are completely inside the image
		// purpose:
		//	- visualization
		//	- needed if the pieces are cropped separately (not used)
		Margin margin = calcOptimalMargin(corners, toPoint2fVec(contour), img.size());

		// 8.2 Compute transformation matrices
		calcProjectionParams(corners, homography, perspective, img.size(), margin);

		// 8.3. Reproject image
		warpPerspective(img, img_r, homography, img.size());
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



		// 10. Piece detection
		std::cout << "Prediction started\n";
		model.predict(img, pred);

		if (pred.num_detections == 0) {
			std::cerr << "No piece was found\n";
			exit(1);
		}

		print_prediction_statistics(pred);
		print_predictions(pred, threshold_score);


		// 11. Filter and transform the prediction
		piece_types.clear();
		piece_scores.clear();
		piece_boxes.clear();

		for (int i = 0; i < pred.num_detections; i++) {
			if (pred.scores[i] < threshold_score)
				continue;

			nr_pieces++;
			piece_boxes.push_back(boundingBoxToRect(pred.boxes[i], imageSize));
			piece_scores.push_back(pred.scores[i]);
			piece_types.push_back(classToPiece(pred.classes[i]));
		}


		// 11. Reproject piece bounding boxes
		piece_boxes_r.clear();
		for (const auto& box : piece_boxes) {
			std::vector<Point2f> bbox_r;
			perspectiveTransformRectangle(box, bbox_r, perspective);
			piece_boxes_r.push_back(bbox_r); 
		}
		show_boundingBoxes("Predicted piece bounding boxes", img_r, piece_boxes_r);


		// 12. Recreate chessboard
		chessboard = recreateChessboard(piece_types, piece_scores, piece_boxes_r, corners);

		// 13. Visualize board
		show_image("Chessboard", getDigitalChessboard(chessboard));

		waitKey();
	}
}


int main()
{
	//Menu menu({
	//	{"Corner detection", testCornerDetection},
	//	{"Print the paths of the calibration images", printCalibrationImagePaths},
	//	{"Camera calibration", testCameraCalibration},
	//	{"Chessboard detection", testChessboardDetection},
	//	{"Chessboard visualization", testVisualizeChessboard},
	//	{"Piece recognition", testPieceRecognition},
	//	{"Chessboard detection and piece recognition", testChessboardDetectionAndPieceRecognition},
	//	});

	//menu.show(std::cin, std::cout);

	testChessboardDetectionAndPieceRecognition();
	return 0;
}