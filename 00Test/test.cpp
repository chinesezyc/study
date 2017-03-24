#include <iostream>
#include <vector>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "../Range.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;
int main(void)
{
	Mat img = imread("../krystal.jpg",0);
	if (!img.data)
		cout << "img is none";
	Mat kk = img(Range(200, 700), Range(100,500));
	imshow("FF", kk);

	//Mat s = img.colRange(0,10);
	//Mat roi = s.rowRange(0, 100);
	//RNG rng((unsigned)time(NULL));

	//rng.fill(roi, RNG::UNIFORM, Scalar(2), Scalar(50));
	//cout << roi<<endl;
	//imshow("roi", roi);
	Mat c;
	for (int i = 0; i < 100; ++i)
		c.push_back(i);
	cout << c << endl;

	/*Mat temp = Mat(img.size(), img.type());
	temp = Scalar::all(50);
	imshow("img", img);
	imshow("temp", img+temp);
*/



/*	Mat_<float> img;
	for (int i : detail_range::Range(3))
		for (int j : detail_range::Range(3))
		{
			img = (Mat_<float>(4, 1) << j,j,i, i);

			cout << img <<endl;
		}
	
	cout << img(0, 0)*/;

	//cv::Size windowSize(60, 60);
	//int step = 50;
	//for(int i=0;i<img.rows-windowSize.height;i += step)
	//	for (int j = 0; j < img.cols - windowSize.width; j += step)
	//	{
	//		cv::Rect rect = cv::Rect(j, i, windowSize.width, windowSize.height);
	//		cv::rectangle(img, rect, cv::Scalar(0,255,0), 1, LINE_AA);
	//		imshow("sliding", img);
	//		waitKey(30);
	//	}
	waitKey(0);

	return 0;
}