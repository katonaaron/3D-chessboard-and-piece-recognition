#include "colors.h"
#include <random>

using namespace cv;

cv::Vec3b convHSVToRGB(float H, float S, float V) {
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

uchar genRandomGrayColor() {
	static std::default_random_engine gen;
	static std::uniform_int_distribution<int> d(0, 255);
	return d(gen);
}

cv::Vec3b genRandomBGRColor() {
	return Vec3b(genRandomGrayColor(), genRandomGrayColor(), genRandomGrayColor());
}

// based on: https://stackoverflow.com/a/1168291
cv::Vec3b genUniqueBGRColor() {
	static const int EXPECTED_MAX = 15;
	static int HUE_FACTOR = 255 / EXPECTED_MAX;
	static int id = 1;

	float hue = (id * HUE_FACTOR) % 255;
	float saturation = 175;
	float brightness = 175;

	id += 7;

	return convHSVToRGB(hue, saturation, brightness);
}

