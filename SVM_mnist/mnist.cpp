#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}


std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

using namespace std;
using namespace cv;
using namespace cv::ml;
#define WIDTH 26
#define HEIGTH 26

vector<string> getFileLine(string filePath);
void getTrainData(Mat &trainData, Mat &trainLabels);


int main(int argc, char* argv)
{
	Mat trainData;

	Mat	trainLabels;
	getTrainData(trainData, trainLabels);
	trainData.convertTo(trainData, CV_32FC1);
	trainLabels.convertTo(trainLabels, CV_32SC1);
	cv::imwrite("trainData.jpg", trainData);
	cv::imwrite("trainLabels.jpg", trainLabels);

	//------------------------SVM----------------------------//
	cv::Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY);
	svm->setDegree(3.0);
	svm->setGamma(3.0);
	svm->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50000, 10));
	cv::Ptr<TrainData> pTrainData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);

	cout << "begin SVM train" << endl;
	//svm->trainAuto(pTrainData);
	//pSVM->trainAuto(TrainData, 10, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE), false);
	svm->train(pTrainData);
	cout << "end SVM train" << endl;

	
svm->save("sss.xml");
	return 0;
}
vector<string> getFileLine(string filePath)
{
	ifstream in(filePath);
	string line;
	vector<string> vfilelabel;
	if (in)
		while (getline(in, line))
		{
			vfilelabel.push_back(line);
		}
	else
		cout << "NO Such File!!" << endl;
	return vfilelabel;
}
void getTrainData(Mat &trainData, Mat &trainLabels)
{
	string dirPath = "C:/Users/O1dCat/Desktop/Tasks/myJob/data/mnist/";
	vector<string> dataList = getFileLine(dirPath + "trainValLabel.txt");

	//ofstream f("te.txt");
	for (string line : dataList)
	{
		//istringstream iss(line);
		//vector<string> str = { istream_iterator<string>{iss}, istream_iterator<string>{} };
		vector<string> str = split(line, ' ');
		//f << dirPath + "mnist/" + str[0] << endl;
		Mat img = imread(dirPath + "mnist/" + str[0], 0);
		img = img.reshape(1, 1);
		trainData.push_back(img);
		trainLabels.push_back(stoi(str[1]));

	}
	//f.close();
}


