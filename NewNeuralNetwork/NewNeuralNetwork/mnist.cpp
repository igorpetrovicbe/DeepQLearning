#pragma once
#include "stdafx.h"
#include "mnist.h"

typedef unsigned char uchar;

using namespace std;

vector<vector<float>> read_mnist_labels(string full_path, int& number_of_labels, int outputSize) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		vector<vector<float>> dataset(number_of_labels);
		for (int i = 0; i < number_of_labels; i++) {
			vector<float> trainExample(outputSize);
			uchar rawExample;
			file.read((char*)&rawExample, 1);

			for (int j = 0; j < outputSize; j++) {
				if (int(rawExample) == j)
					trainExample[j] = 1;
				else
					trainExample[j] = 0;
			}
			dataset[i] = trainExample;
		}
		return dataset;
	}
	else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}

vector<vector<float>> read_mnist_images(string full_path, int& number_of_images, int& image_size) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {

		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		vector<vector<float>> dataset(number_of_images);
		for (int i = 0; i < number_of_images; i++) {
			vector<float> trainExample(image_size);
			uchar* rawExample = new uchar[image_size];
			file.read((char *)rawExample, image_size);
			for (int j = 0; j < image_size; j++)
				trainExample[j] = ((float)rawExample[j]) / 255;
			dataset[i] = trainExample;
		}
		return dataset;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}