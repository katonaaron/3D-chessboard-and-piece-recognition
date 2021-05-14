#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>

enum class Piece
{
	BlackBishop = 0,
	WhiteBishop = 1,
	BlackKing = 2,
	WhiteKing = 3,
	BlackKnight = 4,
	WhiteKnight = 5,
	BlackPawn = 6,
	WhitePawn = 7,
	BlackRook = 8,
	WhiteRook = 9,
	BlackQueen = 10,
	WhiteQueen = 11
};

std::string pieceToString(Piece piece);

const cv::Point2i C_A1(0, 7);
const cv::Point2i C_A2(0, 6);
const cv::Point2i C_A3(0, 5);
const cv::Point2i C_A4(0, 4);
const cv::Point2i C_A5(0, 3);
const cv::Point2i C_A6(0, 2);
const cv::Point2i C_A7(0, 1);
const cv::Point2i C_A8(0, 0);
const cv::Point2i C_B1(1, 7);
const cv::Point2i C_B2(1, 6);
const cv::Point2i C_B3(1, 5);
const cv::Point2i C_B4(1, 4);
const cv::Point2i C_B5(1, 3);
const cv::Point2i C_B6(1, 2);
const cv::Point2i C_B7(1, 1);
const cv::Point2i C_B8(1, 0);
const cv::Point2i C_C1(2, 7);
const cv::Point2i C_C2(2, 6);
const cv::Point2i C_C3(2, 5);
const cv::Point2i C_C4(2, 4);
const cv::Point2i C_C5(2, 3);
const cv::Point2i C_C6(2, 2);
const cv::Point2i C_C7(2, 1);
const cv::Point2i C_C8(2, 0);
const cv::Point2i C_D1(3, 7);
const cv::Point2i C_D2(3, 6);
const cv::Point2i C_D3(3, 5);
const cv::Point2i C_D4(3, 4);
const cv::Point2i C_D5(3, 3);
const cv::Point2i C_D6(3, 2);
const cv::Point2i C_D7(3, 1);
const cv::Point2i C_D8(3, 0);
const cv::Point2i C_E1(4, 7);
const cv::Point2i C_E2(4, 6);
const cv::Point2i C_E3(4, 5);
const cv::Point2i C_E4(4, 4);
const cv::Point2i C_E5(4, 3);
const cv::Point2i C_E6(4, 2);
const cv::Point2i C_E7(4, 1);
const cv::Point2i C_E8(4, 0);
const cv::Point2i C_F1(5, 7);
const cv::Point2i C_F2(5, 6);
const cv::Point2i C_F3(5, 5);
const cv::Point2i C_F4(5, 4);
const cv::Point2i C_F5(5, 3);
const cv::Point2i C_F6(5, 2);
const cv::Point2i C_F7(5, 1);
const cv::Point2i C_F8(5, 0);
const cv::Point2i C_G1(6, 7);
const cv::Point2i C_G2(6, 6);
const cv::Point2i C_G3(6, 5);
const cv::Point2i C_G4(6, 4);
const cv::Point2i C_G5(6, 3);
const cv::Point2i C_G6(6, 2);
const cv::Point2i C_G7(6, 1);
const cv::Point2i C_G8(6, 0);
const cv::Point2i C_H1(7, 7);
const cv::Point2i C_H2(7, 6);
const cv::Point2i C_H3(7, 5);
const cv::Point2i C_H4(7, 4);
const cv::Point2i C_H5(7, 3);
const cv::Point2i C_H6(7, 2);
const cv::Point2i C_H7(7, 1);
const cv::Point2i C_H8(7, 0);

cv::Mat getDigitalChessboard(std::vector<std::pair<Piece, cv::Point2i>> pieces);

#endif // !VISUALIZATION_H

