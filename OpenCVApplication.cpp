// OpenCVApplication.cpp : Defines the entry point for the console application.
//


#include <random>
#include <algorithm>
#include <numeric>

#include "common.h"
#include "Menu.h"
#include "Menu.h"
#include "config.h"
#include "chessboard.h"
#include "util.h"



using namespace cv;

void waitForKey() {
	std::cout << "Press any key...." << std::endl;
	std::cin.get();
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

void adjustBrightness(Mat& img) {
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
	Mat closed;
	morphologyEx(img, closed, MORPH_CLOSE, kernel);
	Mat floatMat;
	img.convertTo(floatMat, CV_32FC1);
	floatMat /= closed;
	normalize(floatMat, floatMat, 0, 255, NORM_MINMAX);
	floatMat.convertTo(img, img.type());
}

uchar genRandomGrayColor() {
	static std::default_random_engine gen;
	static std::uniform_int_distribution<int> d(0, 255);
	return d(gen);
}

Vec3b genRandomBGRColor() {
	return Vec3b(genRandomGrayColor(), genRandomGrayColor(), genRandomGrayColor());
}

Vec3b convHSVToRGB(float H, float S, float V) {
	float C = V * S;
	float h = H / 60;
	float X = (float)(C * (1.0f - fabs(fmod(h, 2) - 1.0f)));
	float m = V - C;

	assert(0 <= h && h <= 6);

	float r1, g1, b1;

	if (h <= 1) {
		r1 = C;
		g1 = X;
		b1 = 0.0f;
	}
	else if (h <= 2) {
		r1 = X;
		g1 = C;
		b1 = 0.0f;
	}
	else if (h <= 3) {
		r1 = 0.0f;
		g1 = C;
		b1 = X;
	}
	else if (h <= 4) {
		r1 = 0.0f;
		g1 = X;
		b1 = C;
	}
	else if (h <= 5) {
		r1 = X;
		g1 = 0.0f;
		b1 = C;
	}
	else {
		r1 = C;
		g1 = 0.0f;
		b1 = X;
	}


	return Vec3b(
		static_cast<uchar>((b1 + m) * 255),
		static_cast<uchar>((g1 + m) * 255),
		static_cast<uchar>((r1 + m) * 255)
	);
}

Vec3b genUniqueBGRColor() {
	static const byte EXPECTED_MAX = 15;
	static int HUE_FACTOR = 255 / EXPECTED_MAX;
	static int id = 1;

	float hue = (id * HUE_FACTOR) % 255;
	float saturation = 175;
	float brightness = 175;

	id += 7;

	return convHSVToRGB(hue, saturation, brightness);
}

int lowThreshold = 50;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

Mat src, src_gray;
Mat dst, detected_edges;

bool isEqual(const Vec4i& _l1, const Vec4i& _l2)
{
	Vec4i l1(_l1), l2(_l2);

	float length1 = sqrtf((l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]));
	float length2 = sqrtf((l2[2] - l2[0]) * (l2[2] - l2[0]) + (l2[3] - l2[1]) * (l2[3] - l2[1]));

	float product = (l1[2] - l1[0]) * (l2[2] - l2[0]) + (l1[3] - l1[1]) * (l2[3] - l2[1]);

	if (fabs(product / (length1 * length2)) < cos(CV_PI / 30))
		return false;

	float mx1 = (l1[0] + l1[2]) * 0.5f;
	float mx2 = (l2[0] + l2[2]) * 0.5f;

	float my1 = (l1[1] + l1[3]) * 0.5f;
	float my2 = (l2[1] + l2[3]) * 0.5f;
	float dist = sqrtf((mx1 - mx2) * (mx1 - mx2) + (my1 - my2) * (my1 - my2));

	if (dist > max(length1, length2) * 0.5f)
		return false;

	return true;
}

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

Vec4i extendedLine(Vec4i line, double d) {
	// oriented left-t-right
	Vec4d _line = line[2] - line[0] < 0 ? Vec4d(line[2], line[3], line[0], line[1]) : Vec4d(line[0], line[1], line[2], line[3]);
	double m = linearParameters(_line)[0];
	// solution of pythagorean theorem and m = yd/xd
	double xd = sqrt(d * d / (m * m + 1));
	double yd = xd * m;
	return Vec4d(_line[0] - xd, _line[1] - yd, _line[2] + xd, _line[3] + yd);
}

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
	if (fabs(a1 - a2) > maxAngleDiff * PI / 180.0) {
		return false;
	}

	// calculate window around extended line
	// at least one point needs to inside extended bounding rectangle of other line,
	std::vector<Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
	return
		pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) == 1 ||
		pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) == 1;
}

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void CannyThreshold(int, void*)
{
	//![reduce_noise]
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));
	//![reduce_noise]

	//![canny]
	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
	//![canny]

	/// Using Canny's output as a mask, we display our result
	//![fill]
	dst = Scalar::all(0);
	//![fill]

	//![copyto]
	src.copyTo(dst, detected_edges);
	//![copyto]

	//![display]
	imshow(window_name, dst);
	//![display]

	const double hough_rho = 1.0;
	const double hough_theta = PI / 180.0;
	const int hough_threshold = 100;
	const int hough_minLineLength = 100;
	const int hough_maxLineGap = 80;

	Mat edges = detected_edges;
	std::vector<Vec4i> lines;
	//Mat result = src.clone();
	Mat result = Mat::zeros(edges.size(), edges.type());

	//Mat binary = edges > 125;  // Convert to binary image

	// Combine similar lines
	int size = 3;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	morphologyEx(edges, edges, MORPH_CLOSE, element);

	imshow("closed edges", edges);


	HoughLinesP(edges, lines, hough_rho, hough_theta, hough_threshold, hough_minLineLength, hough_maxLineGap);

	//std::vector<int> labels;
	//int numberOfLines = partition(lines, labels, isEqual);


	//for (int i = 0; i < lines.rows; i++) {
	//	for (int j = 0; j < lines.cols; j++) {
	//		const Vec4i& lin = lines.at<Vec4i>(i, j);

	//		line(result, Point(lin[0], lin[1]), Point(lin[2], lin[3]), genRandomBGRColor(), 3, LINE_AA);

	//	}
	//}

	for (const auto& lin : lines) {
		line(result, Point(lin[0], lin[1]), Point(lin[2], lin[3]), genRandomBGRColor(), 3, LINE_AA);
	}

	std::cout << lines.size() << "\n";

	size = 15;
	Mat eroded;
	cv::Mat erodeElement = getStructuringElement(MORPH_ELLIPSE, cv::Size(size, size));
	erode(result, result, erodeElement);

	imshow("result", result);
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
	Point2f& r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

inline bool isInside(const Size& imgSize, const Point& point) {
	return 0 <= point.y && point.y < imgSize.height && 0 <= point.x && point.x < imgSize.width;
}

std::vector<Point2i> findIntersections(const std::vector<Vec4i>& lines, Size imageSize) {
	std::vector<Point2i> result;

	for (int i = 0; i < lines.size(); i++) {
		for (int j = 0; j < lines.size(); j++) {
			if (i == j)
				continue;

			const Vec4i& line1 = lines[i];
			const Vec4i& line2 = lines[j];

			Point2i o1 = cv::Point(line1[0], line1[1]);
			Point2i p1 = cv::Point(line1[2], line1[3]);
			Point2i o2 = cv::Point(line2[0], line2[1]);
			Point2i p2 = cv::Point(line2[2], line2[3]);
			Point2f r;

			if (intersection(o1, p1, o2, p2, r) && isInside(imageSize, r)) {
				result.push_back(r);
			}		
		}
	}

	return result;
}

bool isCloseToOtherPoints(const Point2i& point, const std::vector<Point2i>& points, int thresholdDistance) {
	for (const auto& point2 : points) {
		if (norm(point - point2) < thresholdDistance) {
			return true;
		}
	}
	return false;
}

void filterClosePoints(std::vector<Point2i>& points, int thresholdDistance) {
	std::vector<Point2i> savedPoints;

	for (const auto& point : points) {
		if (!isCloseToOtherPoints(point, savedPoints, thresholdDistance)) {
			savedPoints.push_back(point);
		}
	}

	points = std::move(savedPoints);
}

void reprojectImage(const Mat& src, Mat& dst, const std::vector<Point2i>& corners, Size imageSize) {
	static const std::vector<Point2i> dst_points = {
		{0, 0},
		{imageSize.width - 1, 0},
		{imageSize.width - 1, imageSize.height - 1},
		{0, imageSize.height - 1}
	};

	Mat homography = findHomography(corners, dst_points);
	warpPerspective(src, dst, homography, imageSize);
}

void reprojectPoints(const std::vector<Point2f>& src, std::vector<Point2f>& dst, const std::vector<Point2f>& corners, Size imageSize) {
	static const std::vector<Point2f> dst_points = {
		{0, 0},
		{imageSize.width - 1.0f, 0},
		{imageSize.width - 1.0f, imageSize.height - 1.0f},
		{0, imageSize.height - 1.0f}
	};

	Mat pers = getPerspectiveTransform(corners, dst_points);
	perspectiveTransform(src, dst, pers);
}

// 4 points of a retangle
void getFourCorners(const std::vector<Point2f>& corners, Point2f& topLeft, Point2f& bottomLeft, Point2f& topRight, Point2f& bottomRight) {
	topLeft = bottomLeft = topRight = bottomRight = corners[0];

	for (int i = 1; i < corners.size(); i++) {
		const Point& p = corners[i];

		if (p.x <= topLeft.x && p.y <= topLeft.y) {
			topLeft = p;
		}

		if (p.x <= bottomLeft.x && p.y >= bottomLeft.y) {
			bottomLeft = p;
		}

		if (p.x >= topRight.x && p.y <= topRight.y) {
			topRight = p;
		}

		if (p.x >= bottomRight.x && p.y >= bottomRight.y) {
			bottomRight = p;
		}
	}
}


int hough_threshold = 100;

static void CannyThreshold2(int, void*)
{
	const double hough_rho = 1.0;
	const double hough_theta = PI / 180.0;
	//const int hough_threshold = 200;
	const int hough_minLineLength = 100;
	const int hough_maxLineGap = 80;


	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
	imshow("edges before crop", detected_edges);

	std::vector<std::vector<Point2i>> contours;

	findContours(detected_edges, contours, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);
	int largest_contour_index = -1;
	double largest_area = 0;
	Rect bounding_rect;

	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}

	}

	contours = std::vector<std::vector<Point2i>>(1, contours[largest_contour_index]);
	Mat imgWithContours = src.clone();
	drawContours(imgWithContours, contours, -1, Vec3b(0, 0, 255), 1, 8);
	rectangle(imgWithContours, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
	imshow("imgWithContours", imgWithContours);

	int bias = 3;
	bounding_rect.x = max(0, bounding_rect.x - bias);
	bounding_rect.y = max(0, bounding_rect.y - bias);
	bounding_rect.width = min(bounding_rect.width + bias, src.cols - 1 - bounding_rect.x);
	bounding_rect.height = min(bounding_rect.height + bias, src.cols - 1 - bounding_rect.y);

	src = src(bounding_rect);
	src_gray = src_gray(bounding_rect);




	Mat image = src;
	Mat smallerImage = src;
	Mat target = smallerImage.clone();

	namedWindow("Detected Lines", WINDOW_NORMAL);
	namedWindow("Reduced Lines", WINDOW_NORMAL);
	Mat detectedLinesImg = Mat::zeros(target.rows, target.cols, CV_8UC3);
	Mat reducedLinesImg = detectedLinesImg.clone();

	// delect lines in any reasonable way
	Mat grayscale; cvtColor(target, grayscale, COLOR_BGR2GRAY);
	//Ptr<ximgproc::FastLineDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
	std::vector<Vec4i> lines; 
	//detector->detect(grayscale, lines);

	blur(src_gray, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
	imshow(window_name, detected_edges);





	HoughLinesP(detected_edges, lines, hough_rho, hough_theta, hough_threshold, hough_minLineLength, hough_maxLineGap);
	



	// remove small lines
	std::vector<Vec4i> linesWithoutSmall;
	std::copy_if(lines.begin(), lines.end(), std::back_inserter(linesWithoutSmall), [](Vec4f line) {
		float length = sqrtf((line[2] - line[0]) * (line[2] - line[0])
			+ (line[3] - line[1]) * (line[3] - line[1]));
		return length > 30;
		});

	std::cout << "Detected: " << linesWithoutSmall.size() << std::endl;

	// partition via our partitioning function
	std::vector<int> labels;
	int equilavenceClassesCount = cv::partition(linesWithoutSmall, labels, [](const Vec4i l1, const Vec4i l2) {
		return extendedBoundingRectangleLineEquivalence(
			l1, l2,
			// line extension length - as fraction of original line width
			0.2,
			// maximum allowed angle difference for lines to be considered in same equivalence class
			2.0,
			// thickness of bounding rectangle around each line
			10);
		});

	std::cout << "Equivalence classes: " << equilavenceClassesCount << std::endl;

	// grab a random colour for each equivalence class
	RNG rng(215526);
	std::vector<Scalar> colors(equilavenceClassesCount);
	for (int i = 0; i < equilavenceClassesCount; i++) {
		colors[i] = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
	}

	// draw original detected lines
	for (int i = 0; i < linesWithoutSmall.size(); i++) {
		Vec4i& detectedLine = linesWithoutSmall[i];
		line(detectedLinesImg,
			cv::Point(detectedLine[0], detectedLine[1]),
			cv::Point(detectedLine[2], detectedLine[3]), colors[labels[i]], 2);
	}

	// build point clouds out of each equivalence classes
	std::vector<std::vector<Point2i>> pointClouds(equilavenceClassesCount);
	for (int i = 0; i < linesWithoutSmall.size(); i++) {
		Vec4i& detectedLine = linesWithoutSmall[i];
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

	for (size_t i = 0; i < reducedLines.size(); i++) {
		const Vec4i& reduced = reducedLines[i];
		line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), colors[i], 2);
	}

	imshow("Detected Lines", detectedLinesImg);
	imshow("Reduced Lines", reducedLinesImg);
	//waitKey();


	std::vector<Point2i> intersections = findIntersections(reducedLines, src.size());
	std::cout << "nr_intersections: " << intersections.size();

	//auto mnmx = std::minmax_element(intersections.begin(), intersections.end(), [](const auto& a, const auto& b) {
	//	return a.x < b.x || (a.x == b.x && a.y < b.y);
	//	});

	//const int thresholdPointDist = (mnmx.second->x - mnmx.first->x) / 11;
	const int thresholdPointDist = 20;
	filterClosePoints(intersections, thresholdPointDist);

	std::cout << " reduced to: " << intersections.size() << "\n";

	Mat imgWithIntersections = src.clone();
	for (const auto& inter : intersections) {
		circle(imgWithIntersections, inter, 5, Vec3b(0, 0, 255), -1);
	}

	imshow("Intersection points", imgWithIntersections);

	Point2i center = Point2i(src.cols / 2, src.rows / 2);




	std::vector<Point2i> convex_hull;
	convexHull(intersections, convex_hull);

	Mat imgWithConvexHull = src.clone();
	for (const auto& inter : convex_hull) {
		circle(imgWithConvexHull, inter, 5, Vec3b(0, 0, 255), -1);
	}
	imshow("Convex hull of intersection points", imgWithConvexHull);





	if (convex_hull.size() < 4) {
		std::cerr << "the 4 corners were not found";
		exit(1);
	}

	while (convex_hull.size() > 4) {
		const size_t size = convex_hull.size();
		size_t remove_i = -1;
		double min_dist = 0;
		
		for (size_t curr_i = 0; curr_i < size; curr_i++) {

			size_t prev_i = (curr_i - 1 + size) % size;
			size_t next_i = (curr_i + 1) % size;

			Point2i a = convex_hull[prev_i];
			Point2i b = convex_hull[next_i];
			Point2i p = convex_hull[curr_i];

			double ab_len = norm(b - a);
			double dist_p_ab = norm((b - a).cross(a - p)) / norm(b - a);

			if (curr_i == 0 || dist_p_ab < min_dist) {
				min_dist = dist_p_ab;
				remove_i = curr_i;
			}
		}
		convex_hull.erase(convex_hull.begin() + remove_i);
	}

	Mat imgWithIntCorners = src.clone();
	for (const auto& inter : convex_hull) {
		circle(imgWithIntCorners, inter, 5, Vec3b(0, 0, 255), -1);
	}

	imshow("Intersection corners", imgWithIntCorners);


	std::vector<Point2f> corners;
	for (const auto& inter : convex_hull) {
		corners.push_back(inter);
	}

	Mat reprojected;
	std::vector<Point2f> reprojectedCorners;
	reprojectImage(src, reprojected, convex_hull, imageSize);

	
	Mat reprojectedWith4Corners = reprojected.clone();
	reprojectPoints(corners, reprojectedCorners, corners, imageSize);
	for (const auto& corner : reprojectedCorners) {
		circle(reprojectedWith4Corners, corner, 5, Vec3b(0, 0, 255), -1);
	}

	imshow("Reprojected image with four corners", reprojectedWith4Corners);



	//Point2f topLeft, bottomLeft, topRight, bottomRight;
	//getFourCorners(reprojectedCorners, topLeft, bottomLeft, topRight, bottomRight);

	//float dx = (topRight.x - topLeft.x) / 8;
	//float dy = (bottomLeft.y - topLeft.y) / 8;


	//std::vector<Point2f> allReprCorners;
	//for (int i = 0; i <= 8; i++) {
	//	for (int j = 0; j <= 8; j++) {
	//		allReprCorners.push_back(topLeft + Point2f(j * dx, i * dy));
	//	}
	//}

	std::vector<Point2f> allReprCorners;
	std::vector<Point2f> intersectionsf;
	for (const auto& inter : intersections) {
		intersectionsf.push_back(inter);
	}

	reprojectPoints(intersectionsf, allReprCorners, corners, imageSize);


	Mat reprojectedWithAllCorners = reprojected.clone();
	for (const auto& corner : allReprCorners) {
		circle(reprojectedWithAllCorners, corner, 5, Vec3b(0, 0, 255), -1);
	}
	imshow("Reprojected image with all corners", reprojectedWithAllCorners);

}



void testChessboardDetection() {

	const double canny_threshold1 = 50;
	const double canny_threshold2 = 150;
	const int canny_appertureSize = 3;

	const double hough_rho = 1.0;
	const double hough_theta = PI / 180.0;
	//const int hough_threshold = 100;
	const int hough_minLineLength = 100;
	const int hough_maxLineGap = 80;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_COLOR);
		Mat resized;
		resize(img, resized, imageSize);
		Mat result = resized.clone();
		src = resized.clone();

		Mat imgGray;
		cvtColor(resized, imgGray, COLOR_BGR2GRAY);
		src_gray = imgGray.clone();

		imshow("image", imgGray);

		namedWindow(window_name, WINDOW_AUTOSIZE);

		//![create_trackbar]
/// Create a Trackbar for user to enter threshold
		//createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold2);
		createTrackbar("Hough Threshold:", window_name, &hough_threshold, 500, CannyThreshold2);
		//![create_trackbar]

		/// Show the image
		CannyThreshold2(0, 0);

		/// Wait until user exit program by pressing a key
		waitKey(0);

		/*Mat edges;
		blur(imgGray, edges, Size(3, 3));
		Canny(edges, edges, canny_threshold1, canny_threshold2, canny_appertureSize);

		imshow("edges", edges);

		Mat lines;
		HoughLinesP(edges, lines, hough_rho, hough_theta, hough_threshold, hough_minLineLength, hough_maxLineGap);

		for (int i = 0; i < lines.rows; i++) {
			for (int j = 0; j < lines.cols; j++) {
				const Vec4i& lin = lines.at<Vec4i>(i, j);

				line(result, Point(lin[0], lin[1]), Point(lin[2], lin[3]), Vec3b(0, 0, 255), 3, LINE_AA);

			}
		}

		imshow("result", result);*/

		//imshow("lines", lines);


	/*	adjustBrightness(imgGray);

		imshow("image2", imgGray);

		std::vector<Point2f> corners;

		bool found = detectCorners(imgGray, boardSize, winSize, corners);

		waitKey();

		if (!found) {
			std::cerr << "Corners were not found\n";
			exit(1);
		}

		drawChessboardCorners(imgGray, boardSize, Mat(corners), found);*/

	
		//imwrite("result.png", imgGray);

		waitKey();
	}
}

int main()
{
	Menu menu({
		{"Corner detection", testCornerDetection},
		{"Print the paths of the calibration images", printCalibrationImagePaths},
		{"Camera calibration", testCameraCalibration},
		{"Chessboard detection", testChessboardDetection}
		});

	menu.show(std::cin, std::cout);
	return 0;
}