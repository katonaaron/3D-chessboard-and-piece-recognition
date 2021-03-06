#pragma once

#include <opencv2/opencv.hpp>

struct Config {
	const std::string path_model_graph = "model\\10000\\frozen_inference_graph.pb";

	const cv::Size imageSize = cv::Size(512, 512);
} const config;

//const Size boardSize = Size(9, 6);
const cv::Size boardSize = cv::Size(8, 8);
const cv::Size winSize = cv::Size(11, 11);
const cv::Size calibImgSize = cv::Size(1, 1);
const cv::Size imageSize(512, 512);
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
