#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

typedef void (*Routine)();

class Menu
{
public: 
	Menu(std::vector<std::pair<std::string, Routine>> options) : options(std::move(options)) {}

	void show(std::istream& is, std::ostream& os);

private:
	std::vector<std::pair<std::string, Routine>> options;
};

