#pragma once

#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>


/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, const int* hist, const int  hist_cols, const int hist_height);

std::vector<std::string> getFilesInDir(const std::string& dirName);
