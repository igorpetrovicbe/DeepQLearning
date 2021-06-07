// NewNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Network.h"


typedef unsigned char uchar;//TODO: Pomeriti dataset u posebnu klasu?

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

vector<vector<float>> xorInputs(vector<vector<float>>& inputs) {
	inputs[0] = { 0, 0 };
	inputs[1] = { 0, 1 };
	inputs[2] = { 1, 0 };
	inputs[3] = { 1, 1 };
	return inputs;
}

vector<vector<float>> xorOutputs(vector<vector<float>>& outputs) {
	outputs[0] = { 0 };
	outputs[1] = { 1 };
	outputs[2] = { 1 };
	outputs[3] = { 0 };
	return outputs;
}

int main()
{
	srand(time(NULL));

	int inputSize = 784;
	int outputSize = 10;
	int hiddenSize = 500;
	int hiddenNumber = 2;
	float learningRate = 0.02;
	Network net(inputSize, hiddenSize, hiddenNumber, outputSize, learningRate);

	omp_set_num_threads(2);

	int maxIterations = 500000;
	int batchSize = 12;

	int datasetSize;

	vector<vector<float>>datasetInputs = read_mnist_images("train-images.idx3-ubyte", datasetSize, inputSize);
	vector<vector<float>>datasetOutputs = read_mnist_labels("train-labels.idx1-ubyte", datasetSize, outputSize);
	
	int validationSetSize = 10000;
	int trainingSetSize = datasetSize - validationSetSize;

	//Train
	int epoch = 1;
	double startTime = omp_get_wtime();
	double endTime;
	for (int i = 0; i < maxIterations; i++) {
		net.Fit(datasetInputs[i % trainingSetSize], datasetOutputs[i % trainingSetSize]);
		if (i % batchSize == 0 && i != 0)
			net.UpdateWeights();

		if (i % 1000 == 0) {
			endTime = omp_get_wtime();
			cout << i << "\tTook: " << (endTime - startTime) * 1000 << "ms." << endl;
			startTime = omp_get_wtime();
		}
		if (i % trainingSetSize == trainingSetSize - 1) {
			cout << "Epoch" << epoch << " Loss: " << net.resetLoss() / trainingSetSize << endl;
			epoch++;
			net.setLearningRate(learningRate / sqrt(epoch));
		}
	}
	endTime = omp_get_wtime();
	cout << endl << "\tTest finished. Took: " << (endTime - startTime) * 1000 << "ms." << endl << endl;

	//Calculate Accurracy
	vector<float> output(outputSize);
	int correct = 0;
	for (int k = trainingSetSize; k < datasetSize; k++) {
		output = net.Propagate(datasetInputs[k]);
		int choice = 0;
		for (int i = 1; i < outputSize; i++)
			if (output[i] > output[choice])
				choice = i;
		if (datasetOutputs[k][choice] == 1)
			correct++;
	}
	cout << "Accurracy: " << 100 * ((float)correct) / validationSetSize << "%" << endl;
	std::system("pause");

	//Calculate Accurracy on training set (for comparison)
	correct = 0;
	for (int k = 0; k < validationSetSize; k++) {
		output = net.Propagate(datasetInputs[k]);
		int choice = 0;
		for (int i = 1; i < outputSize; i++)
			if (output[i] > output[choice])
				choice = i;
		if (datasetOutputs[k][choice] == 1)
			correct++;
	}
	cout << "Accurracy on TS: " << 100 * correct / validationSetSize << "%" << endl;
	std::system("pause");

	//Print Test
	int testSize = 10;
	for (int i = 0; i < testSize; i++) {
		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++) {
				if (datasetInputs[i][j * 28 + k] > 0.5)
					cout << '#';
				else
					cout << ' ';
			}
			cout << endl;
		}
		output = net.Propagate(datasetInputs[i]);
		cout << '(';
		for (int j = 0; j < outputSize; j++) {
			cout << output[j] << ", ";
		}
		cout << ')' << endl << endl;
	}

	/* XOR TEST
	int testSize = 5;
	vector<float> output(outputSize);
	for (int i = 0; i < testSize; i++) {
		output = net.Propagate(datasetInputs[i]);
		for (int j = 0; j < outputSize; j++) {
			cout << output[j] << ", ";
		}
		cout << endl;
	}
	*/
	
	std::system("pause");

    return 0;
}

