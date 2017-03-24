#include <iostream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
static int STEP;


Mat traverse1(const Mat& src);
Mat traverse2(const Mat& src);
Mat traverse3(const Mat& src);
Mat traverse4(const Mat& src);

int main(void)
{
	Mat img = imread("../krystal.jpg", 0);
	double t = (double)getTickCount();
	for (int K = 1; K <= 8; ++K)
	{
		STEP = 256 / (1 << K);
		string str;
		Mat traveImg = traverse1(img);
		cv::imwrite("k1" + to_string(K) + ".jpg", traveImg);
	}
	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	cout << "Times" << t << endl;
	cv::waitKey(0);
	return 0;
}

Mat traverse1(const Mat& src)
{
	switch (src.channels())
	{
	case 1:
	{
		/*Mat_<uchar> dst = Mat(src.size(), src.type());
		for (int i = 0; i < src.rows; ++i)
			for (int j = 0; j < src.cols; ++j)
			{
				int val = src.at<uchar>(i, j);
				int step = STEP;
				while (val > step)
					step += STEP;
				dst(i, j) = step;
			}*/
		Mat_<uchar> dst = Mat(src.size(), src.type());
		uchar table[256] = {};
		for (int i = 0; i < 256; ++i)
			table[i] = STEP*(i / STEP);
		for (int i = 0; i < src.rows; ++i)
			for (int j = 0; j < src.cols; ++j)
				dst(i, j) = table[src.at<uchar>(i, j)];
		return dst;
		break;
	}


	case 3:
	{
		Mat dst = Mat(src.size(), src.type());
		for (int i = 0; i < src.rows; ++i)
			for (int j = 0; j < src.cols; ++j)
			{
				int bVal = src.at<cv::Vec3b>(i, j)[0];
				int gVal = src.at<cv::Vec3b>(i, j)[1];
				int rVal = src.at<cv::Vec3b>(i, j)[2];

				int step = STEP;
				while (bVal > step)
					step += STEP;
				dst.at<cv::Vec3b>(i, j)[0] = step;

				step = STEP;
				while (gVal > step)
					step += STEP;
				dst.at<cv::Vec3b>(i, j)[1] = step;

				step = STEP;
				while (rVal > step)
					step += STEP;
				dst.at<cv::Vec3b>(i, j)[2] = step;
				/*
				for (int step = STEP; bVal > step && gVal > step && rVal > step; step += STEP)
				{
					if (step > bVal && (step - bVal) / STEP == 0)
						dst.at<cv::Vec3b>(i, j)[0] = step;
					if (step > gVal && (step - gVal) / STEP == 0)
						dst.at<cv::Vec3b>(i, j)[1] = step;
					if (step > rVal && (step - rVal) / STEP == 0)
						dst.at<cv::Vec3b>(i, j)[2] = step;
				}*/
			}
		return dst;
		break;
	}

	default:
		break;
	}
}
Mat traverse2(const Mat& src)
{
	switch (src.channels())
	{
	case 1:
	{
		Mat dst = src.clone();
		auto it = dst.begin<uchar>();
		auto itend = dst.end<uchar>();
		for (; it != itend; it++)
		{
			int val = *it;
			//cout << val << " ";
			int step = STEP;
			while (val > step)
				step += STEP;
			*it = step;
			//cout << int(*it) << endl;
		}
		return dst;
		break;
	}
	case 3:
	{
		Mat dst = src.clone();
		auto it = dst.begin<cv::Vec3b>();
		auto itend = dst.end<cv::Vec3b>();

		for (; it != itend; it++)
		{
			int step = STEP;
			int bVal = (*it)[0];
			while (bVal > step)
				step += STEP;
			(*it)[0] = step;

			step = STEP;
			int gVal = (*it)[1];
			while (gVal > step)
				step += STEP;
			(*it)[1] = step;

			step = STEP;
			int rVal = (*it)[2];
			while (rVal > step)
				step += STEP;
			(*it)[2] = step;
		}
		return dst;
		break;
	}
	default:
		break;
	}


}

Mat traverse3(const Mat& src)
{
	switch (src.channels())
	{
	case 1:
	{
		Mat dst = src.clone();
		int rowNumber = dst.rows;
		int colNumber = dst.cols*dst.channels();
		for (int i = 0; i < rowNumber; ++i)
		{
			uchar* data = dst.ptr<uchar>(i);
			for (int j = 0; j < colNumber; ++j)
			{
				int val = data[j];
				int step = STEP;
				while (val > step)
					step += STEP;
				data[j] = step;
			}
		}
		return dst;
		break;
	}
	case 3:
	{
		/*Mat dst = src.clone();
		int rowNumber = dst.rows;
		int colNumber = dst.cols;
		for (int i = 0; i < rowNumber; ++i)
		{
			cv::Vec3b* data = dst.ptr<cv::Vec3b>(i);
			for (int j = 0; j < colNumber; ++j)
			{
				int step = STEP;
				int bval = data[j][0];
				while (bval > step)
					step += STEP;
				data[j][0] = step;

				step = STEP;
				int gval = data[j][1];
				while (gval > step)
					step += STEP;
				data[j][1] = step;

				step = STEP;
				int rval = data[j][2];
				while (rval > step)
					step += STEP;
				data[j][2] = step;
			}*/
		Mat dst = src.clone();
		int rowNumber = dst.rows;
		int colNumber = dst.cols*dst.channels();
		for (int i = 0; i < rowNumber; ++i)
		{
			uchar* data = dst.ptr<uchar>(i);
			for (int j = 0; j < colNumber; ++j)
			{
				int step = STEP;
				int bval = data[j];
				while (bval > step)
					step += STEP;
				data[j] = step;
			}
		}
		return dst;
		break;
	}
	default:
		break;
	}



}

Mat traverse4(const Mat& src)
{
	Mat_<uchar> lookUpTable = Mat(1, 256, CV_8U);

	for (int i = 0; i < 256; ++i)
	{
		lookUpTable(i) = STEP*(i / STEP);
	}

	Mat dst = Mat(src.size(), src.type());
	cv::LUT(src, lookUpTable, dst);
	return dst;

}