#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
#define WINDOW_NAME "线性混合示例"
const int g_nmaxAlphaValue = 100;
int g_nAlphaValueSilder;
double g_dAlphaValue, g_dBetaValue;
Mat g_srcImg1;
Mat g_srcImg2;
Mat g_dstImg;
void onTrackBar(int, void*);

int main(int argc, char** argv)
{
	g_srcImg1 = imread("../krystal.jpg");
	g_srcImg1.copyTo(g_srcImg2);
	cv::flip(g_srcImg2, g_srcImg2, -1);
	g_nAlphaValueSilder = 70;
	cv::namedWindow(WINDOW_NAME, 1);
	string TrackBarName = "透明值";
	TrackBarName += to_string(g_nmaxAlphaValue);
	cout << TrackBarName << g_nAlphaValueSilder;
	cv::createTrackbar(TrackBarName, WINDOW_NAME, &g_nAlphaValueSilder, g_nmaxAlphaValue, onTrackBar);
	for(int i=0;i<10;++i)
		cout << rand()%3 + 1 << endl;
	imshow("g_srcImg2", g_srcImg2);
	imshow("g_srcImg1", g_srcImg1);
	cvWaitKey(0);
	return 0;
}

void onTrackBar(int, void*)
{
	g_dAlphaValue = (double)g_nAlphaValueSilder / g_nmaxAlphaValue;
	g_dBetaValue = (1.0 - g_dAlphaValue);
	addWeighted(g_srcImg1, g_dAlphaValue, g_srcImg2, g_dBetaValue, 0.0, g_dstImg);

	imshow(WINDOW_NAME, g_dstImg);
}
