#pragma once

#include <opencv2/opencv.hpp>

//const Size boardSize = Size(9, 6);
const Size boardSize = Size(8, 8);
const Size winSize = Size(11, 11);
const Size calibImgSize = Size(1, 1);
const Size imageSize(512, 512);
const float squareSize = 50; 

const std::string calibrationImageDir = "images\\left";

const std::string path_visual_dir = "images\\visualization\\png\\";
const std::string ext_visual = ".png";
const std::string path_piece_black_bishop = path_visual_dir + "black-bishop" + ext_visual;
const std::string path_piece_white_bishop = path_visual_dir + "white-bishop" + ext_visual;
const std::string path_piece_black_king = path_visual_dir + "black-king" + ext_visual;
const std::string path_piece_white_king = path_visual_dir + "white-king" + ext_visual;
const std::string path_piece_black_knight = path_visual_dir + "black-knight" + ext_visual;
const std::string path_piece_white_knight = path_visual_dir + "white-knight" + ext_visual;
const std::string path_piece_black_pawn = path_visual_dir + "black-pawn" + ext_visual;
const std::string path_piece_white_pawn = path_visual_dir + "white-pawn" + ext_visual;
const std::string path_piece_black_rook = path_visual_dir + "black-rook" + ext_visual;
const std::string path_piece_white_rook = path_visual_dir + "white-rook" + ext_visual;
const std::string path_piece_black_queen = path_visual_dir + "black-queen" + ext_visual;
const std::string path_piece_white_queen = path_visual_dir + "white-queen" + ext_visual;
const std::string path_board = path_visual_dir + "board" + ext_visual;
