#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#define NTRAINING_SAMPLES   100         // Number of training samples per class
#define FRAC_LINEAR_SEP     0.9f        // Fraction of samples which compose the linear separable part
using namespace cv;
using namespace cv::ml;
using namespace std;

int main()
{
	// Data for visual representation
	const int WIDTH = 512, HEIGHT = 800;
	Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
;
	//--------------------- 1. Set up training data randomly ---------------------------------------
	Mat trainData(2 * NTRAINING_SAMPLES, 2, CV_32FC1);
	Mat labels(2 * NTRAINING_SAMPLES, 1, CV_32SC1);
	RNG rng(100); // Random value generation class
				  // Set up the linearly separable part of the training data
	int nLinearSamples = (int)(FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

	cout << labels.type() << endl;
	// Generate random points for the class 1
	//��ΪrowRange�������ص�Mat��ָ�룬���Կ���ͨ�����ص�ָ��ı�ԭʼMat����ֵ
	//ȡtrainData��ǰ90�У���ʱtrainClass��sizeΪ90��2�У�����Ϊ��Ч
	Mat trainClass = trainData.rowRange(0, nLinearSamples);
	// The x coordinate of the points is in [0, 0.4)
	//c��size 90��1��
	Mat c = trainClass.colRange(0, 1);
	//��0-244.8���������������c
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));


	// Generate random points for the class 2
	trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
	// The x coordinate of the points is in [0.6, 1]
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));



	//------------------ Set up the non-linearly separable part of the training data ---------------
	// Generate random points for the classes 1 and 2
	trainClass = trainData.rowRange(nLinearSamples, 2 * NTRAINING_SAMPLES - nLinearSamples);
	// The x coordinate of the points is in [0.4, 0.6)
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
	//------------------------- Set up the labels for the classes ---------------------------------
	labels.rowRange(0, NTRAINING_SAMPLES).setTo(1);  // Class 1
	labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(2);  // Class 2



	//------------------------ 2. Set up the support vector machines parameters --------------------
	//------------------------ 3. Train the svm ----------------------------------------------------
	cout << "Starting training process" << endl;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.1);
	svm->setKernel(SVM::POLY);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
	svm->train(trainData, ROW_SAMPLE, labels);

	//Ptr<SVM> svm = SVM::load("svm.xml");
	cout << "Finished training process" << endl;
	svm->save("sss.xml");
	//------------------------ 4. Show the decision regions ----------------------------------------
	Vec3b green(100, 100, 0), blue(0, 100, 100);
	for (int i = 0; i < I.rows; ++i)//800
		for (int j = 0; j < I.cols; ++j)//512
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j,i);
			float response = svm->predict(sampleMat);
			if (response == 1)    I.at<Vec3b>(i, j) = green;
			else if (response == 2)    I.at<Vec3b>(i, j) = blue;
		}
	//----------------------- 5. Show the training data --------------------------------------------
	int thick = -1;
	int lineType = 8;
	float px, py;
	// Class 1
	for (int i = 0; i < NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, Point((int)px, (int)py), 3, Scalar(0, 255, 0), thick, lineType);
	}
	// Class 2
	for (int i = NTRAINING_SAMPLES; i <2 * NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, Point((int)px, (int)py), 3, Scalar(255, 0, 0), thick, lineType);
	}
	//------------------------- 6. Show support vectors --------------------------------------------
	thick = 2;
	lineType = 8;
	Mat sv = svm->getUncompressedSupportVectors();
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		circle(I, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thick, lineType);
	}
	imwrite("result2.png", I);                      // save the Image
	imshow("SVM for Non-Linear Training Data", I); // show it to the user
	waitKey(0);
}