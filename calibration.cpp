#include "calibration.h"

using namespace cv;

bool detectCorners(const cv::Mat& img, cv::Size boardSize, cv::Size winSize, std::vector<cv::Point2f>& corners)
{
	const int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
	const TermCriteria termCriteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);

	bool found = findChessboardCorners(img, boardSize, corners, chessBoardFlags);

	if (!found)
		return false;

	Mat imgGray;
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	cornerSubPix(imgGray, corners, winSize, Size(-1, -1), termCriteria);

	return true;
}

void detectCorners(const std::vector<std::string>& imagePaths, cv::Size boardSize, cv::Size winSize, std::vector<std::vector<cv::Point2f>>& corners)
{
	corners.clear();

	for (const auto& path : imagePaths) {
		Mat img = imread(path, IMREAD_COLOR);
		std::vector<cv::Point2f> imgCorners;

		bool found = detectCorners(img, boardSize, winSize, imgCorners);

		if (found)
			corners.push_back(imgCorners);
	}

}

void calcBoardCornerPositions(Size boardSize, float squareSize, std::vector<Point3f>& corners)
{
	corners.clear();

	for (int i = 0; i < boardSize.height; ++i)
		for (int j = 0; j < boardSize.width; ++j)
			corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
}

bool calibrateCamera(const std::vector<std::vector<cv::Point2f>>& imgPoints, cv::Size imageSize, std::vector<cv::Point3f>& objPoints, 
	cv::Mat& cameraMatrix, cv::Mat& distCoeffs, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs)
{

	std::vector<std::vector<Point3f>> objectPoints(imgPoints.size(), objPoints);

	const int iFixedPoint = -1;
	const int flags = CALIB_USE_LU;

	double rms = calibrateCameraRO(objectPoints, imgPoints, imageSize, iFixedPoint,
		cameraMatrix, distCoeffs, rvecs, tvecs, objPoints, flags);

	std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;

	return checkRange(cameraMatrix) && checkRange(distCoeffs);
}
