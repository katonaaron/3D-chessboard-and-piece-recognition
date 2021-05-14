#include "stdafx.h"
#include "tensorflow.h"

using namespace cv;
using namespace tensorflow;

tensorflow::Tensor matToTensor(const cv::Mat& mat) {
	int height = mat.rows;
	int width = mat.cols;
	int depth = mat.channels();
	Tensor inputTensor(tensorflow::DT_UINT8, tensorflow::TensorShape({ 1, height, width, depth }));
	auto inputTensorMapped = inputTensor.tensor<tensorflow::uint8, 4>();

	cv::Mat frame;
	mat.convertTo(frame, CV_8UC3);
	const tensorflow::uint8* source_data = (tensorflow::uint8*) frame.data;
	for (int y = 0; y < height; y++) {
		const tensorflow::uint8* source_row = source_data + (y * width * depth);
		for (int x = 0; x < width; x++) {
			const tensorflow::uint8* source_pixel = source_row + (x * depth);
			for (int c = 0; c < depth; c++) {
				const tensorflow::uint8* source_value = source_pixel + c;
				inputTensorMapped(0, y, x, c) = *source_value;
			}
		}
	}
	return inputTensor;
}

Model::Model(std::string path_to_graph)
{
	// Initialize a tensorflow session

	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		exit(1);
	}

	// Read in the protobuf graph we exported
	// (The path seems to be relative to the cwd. Keep this in mind
	// when using `bazel run` since the cwd isn't where you call
	// `bazel run` but from inside a temp folder.)
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), path_to_graph, &graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		exit(1);
	}

	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		exit(1);
	}
}

void Model::predict(const cv::Mat& image, Prediction& pred)
{
	make_prediction(matToTensor(image), pred);
}

void Model::make_prediction(Tensor& imageTensor, Prediction& pred)
{
	const std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "image_tensor", imageTensor },
	};

	const std::vector<std::string> output_tensor_names = {
		"num_detections",
		"detection_boxes",
		"detection_classes",
		"detection_scores"
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our operation from the graph
	Status status = session->Run(inputs, output_tensor_names, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		exit(1);
	}

	auto predicted_num_detections = outputs[0].tensor<float, 1>();
	auto predicted_boxes = outputs[1].tensor<float, 3>();
	auto predicted_classes = outputs[2].tensor<float, 2>();
	auto predicted_scores = outputs[3].tensor<float, 2>();

	pred.num_detections = static_cast<int>(predicted_num_detections(0));

	for (int i = 0; i < pred.num_detections; i++) {
		std::vector<float> coords;
		for (int j = 0; j < 4; j++) {
			coords.push_back(predicted_boxes(0, i, j));
		}
		pred.boxes.push_back(coords);
		pred.classes.push_back(static_cast<int>(predicted_classes(0, i)));
		pred.scores.push_back(predicted_scores(0, i));
	}
}
