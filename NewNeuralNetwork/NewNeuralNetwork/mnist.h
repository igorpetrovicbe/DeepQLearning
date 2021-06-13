#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <time.h>
#include <conio.h>
#include <iomanip>
#include <omp.h>

using namespace std;

vector<vector<float>> read_mnist_labels(string full_path, int& number_of_labels, int outputSize);
vector<vector<float>> read_mnist_images(string full_path, int& number_of_images, int& image_size);