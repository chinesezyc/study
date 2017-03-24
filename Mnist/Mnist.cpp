#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>


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
	
	//trainData = trainData.colRange(1, 3);
	//int labels[4] = { 1,2,3,4 };
	//float train[4][2] = { { 50,50 },{ 50,450 },{ 450,450 },{ 450,50 } };
	////float trainData[4][2] = { { 100, 10 },{ 10, 500 },{ 500, 10 },{ 500, 500 } };

	//Mat trainData = Mat(4, 2, CV_32FC1, train);
	//Mat trainLabels = Mat(4, 1, CV_32SC1, labels);

	//cv::imwrite("trainData.jpg", trainData);
	//cv::imwrite("trainLabels.jpg", trainLabels);
	//for (int i = 0; i < 20; ++i)
	//	cout << trainData.row(i).cols << endl;
	//cout << trainData.size() << "     " << trainLabels.size() << endl;
	//cout << trainData << endl;
	//cout << trainLabels << endl;
	//------------------------SVM----------------------------//
	cv::Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	//svm->setKernel(SVM::POLY);
	//svm->setDegree(3.0);
	//svm->setGamma(0.01);
	//svm->setC(10.0);

	svm->setKernel(SVM::RBF);
	//    //svm->setDegree(10.0);
	svm->setGamma(0.01);
	//    //svm->setCoef0(1.0);
	svm->setC(10.0);
	//    //svm->setNu(0.5);
	//    //svm->setP(0.1);


	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER||TermCriteria::EPS, 10000, FLT_EPSILON));
	cv::Ptr<TrainData> pTrainData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);


	cout << "begin SVM train" << endl;
	//svm->trainAuto(pTrainData);
	//svm->trainAuto(pTrainData, 10, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE), false);
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	cout << "end SVM train" << endl;
	//-----------------------SAVE---------------
	svm->save("abc_RBF.xml");
	//----------------------TEST----------------

	/*cv::Ptr<SVM> svm = SVM::load("abc_RBF.xml");
	Mat testData;
	Mat testLabels;
	getTrainData(testData, testLabels);
	testData.convertTo(testData, CV_32FC1);
	testLabels.convertTo(testLabels, CV_32SC1);

	cout << testData.rows << "     " << testLabels.rows << endl;

	float currentCount = 0;
	for (int i = 10; i < testData.rows; ++i)
	{
		Mat sample = testData.row(i);
		float res = svm->predict(sample);
		cout << "Response: " << res << "    " << "Label: " << testLabels.at<int>(i, 0) << endl;
		res = std::abs(res - testLabels.at<int>(i, 0)) <= 10 ? 1.0f : 0.0f;
		currentCount += res;

	}
	cout << "正确识别的图片数量为：" << currentCount << endl;
	cout << "错误率为：" << (testData.rows - currentCount) / testData.rows * 100 << "%" << endl;*/
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
	//-------------------------TRAIN------------------------------------
	vector<string> dataList = getFileLine(dirPath + "trainValLabel.txt");

	//-------------------------TRAIN------------------------------------
	//vector<string> dataList = getFileLine(dirPath + "testLabel.txt");

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


