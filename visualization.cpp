#include "visualization.h"
#include "config.h"


std::string getPieceFilePath(Piece piece)
{
	switch (piece) {
	case Piece::BlackBishop:
		return path_piece_black_bishop;
	case Piece::WhiteBishop:
		return path_piece_white_bishop;
	case Piece::BlackKing:
		return path_piece_black_king;
	case Piece::WhiteKing:
		return path_piece_white_king;
	case Piece::BlackKnight:
		return path_piece_black_knight;
	case Piece::WhiteKnight:
		return path_piece_white_knight;
	case Piece::BlackPawn:
		return path_piece_black_pawn;
	case Piece::WhitePawn:
		return path_piece_white_pawn;
	case Piece::BlackRook:
		return path_piece_black_rook;
	case Piece::WhiteRook:
		return path_piece_white_rook;
	case Piece::BlackQueen:
		return path_piece_black_queen;
	case Piece::WhiteQueen:
		return path_piece_white_queen;
	}
	return "";
}

Mat getDigitalChessboard(std::vector<std::pair<Piece, Point2i>> pieces)
{
	const int blockSize = 64;
	const int imageSize = blockSize * 8;

	Mat board = imread(path_board, IMREAD_COLOR);
	Mat digitalChessboard(imageSize, imageSize, CV_8UC4, Scalar::all(0));

	int row = 0;
	for (int i = 0; i < imageSize; i = i + blockSize) {
		int col = 0;
		for (int j = 0; j < imageSize; j = j + blockSize) {
			Mat ROI = board(Rect(i, j, blockSize, blockSize));
			for (int k = 0; k < pieces.size(); k++)
			{
				if (pieces.at(k).second.x == row && pieces.at(k).second.y == col)
				{
					std::string path = getPieceFilePath(pieces.at(k).first);
					Mat piece = imread(path, IMREAD_UNCHANGED);
					for (int i = 0; i < piece.rows; i++) {
						for (int j = 0; j < piece.cols; j++) {
							Vec4b pixel = piece.at<Vec4b>(i, j);
							if (pixel[3] != 0) {
								ROI.at<Vec3b>(i + 10, j + 10) = Vec3b(pixel[0], pixel[1], pixel[2]);
							}
						}
					}
				}
			}
			col++;
		}
		row++;
	}
	return board;

}