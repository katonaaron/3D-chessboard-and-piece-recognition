#ifndef TENSORFLOW_H_INCLUDED
#define TENSORFLOW_H_INCLUDED

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include <opencv2/opencv.hpp>

tensorflow::Tensor matToTensor(const cv::Mat& mat);

struct Prediction {
	int num_detections;
	std::vector<std::vector<float>> boxes;
	std::vector<int> classes;
	std::vector<float> scores;
};

class Model {
public:
	Model(std::string path_to_graph);
	
	void predict(const cv::Mat& input_image, Prediction& pred);


private:
	tensorflow::Session* session = NULL;

	void make_prediction(tensorflow::Tensor& input_image, Prediction& pred);
};

#endif // !TENSORFLOW_H_INCLUDED
