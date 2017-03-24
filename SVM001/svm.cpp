#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;
int main(void)
{
	int width = 512, height = 512;
	Mat image = Mat::zeros(width, height, CV_8UC3);
	int labels[4] = { 1,2,3,4 };
	float trainData[4][2] = { {50,50},{50,450},{450,450},{450,50} };
	//float trainData[4][2] = { { 100, 10 },{ 10, 500 },{ 500, 10 },{ 500, 500 } };

	Mat trainMat = Mat(4, 2, CV_32FC1, trainData);
	Mat labelMat = Mat(4, 1, CV_32SC1, labels);

	//creat SVM
	cv::Ptr<cv::ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::POLY);
	svm->setDegree(1.0);

	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	svm->train(trainMat, ml::ROW_SAMPLE, labelMat);

	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), yellow(0, 255, 255);

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);
			double ratio = 0.5;
			if (response == 1)
				image.at<Vec3b>(i, j) = green*ratio;
			else if (response == 2)
				image.at<Vec3b>(i, j) = blue*ratio;
			else if (response == 3) {
				image.at<Vec3b>(i, j) = red*ratio;
			}
			else if (response == 4) {
				image.at<Vec3b>(i, j) = yellow*ratio;
			}
		}

	}
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(50, 50), 5, Scalar(0, 255, 0), thickness, lineType);
	circle(image, Point(50, 450), 5, Scalar(255, 0, 0), thickness, lineType);
	circle(image, Point(450, 450), 5, Scalar(0, 0, 255), thickness, lineType);
	circle(image, Point(450, 50), 5, Scalar(0, 255, 255), thickness, lineType); 
	/*circle(image, Point(100, 10), 5, Scalar(0, 255, 0), thickness, lineType);
	circle(image, Point(10, 500), 5, Scalar(255, 0, 0), thickness, lineType);
	circle(image, Point(500, 10), 5, Scalar(0, 0, 255), thickness, lineType);
	circle(image, Point(500, 500), 5, Scalar(0, 255, 255), thickness, lineType);*/

	thickness = 2;
	lineType = 8;
	Mat sv = svm->getSupportVectors();
	std::cout << sv << std::endl;

	for (int i = 0; i < sv.rows; ++i) {
		const float* v = sv.ptr<float>(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, CV_RGB(128, 128, 128), 2);
	}
	//imwrite("result.png", image);        // save the image
	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);

	return 0;
}