// OpenCVApplication.cpp : Defines the entry point for the console application.
//

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

int main()
{
	Menu menu({
		{"Corner detection", testCornerDetection},
		{"Print the paths of the calibration images", printCalibrationImagePaths},
		{"Camera calibration", testCameraCalibration}
		});

	menu.show(std::cin, std::cout);
	return 0;
}