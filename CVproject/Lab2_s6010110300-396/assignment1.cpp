
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "TrackingObjectSet.h"
#include "Tracker.h"
#include <opencv2/videoio.hpp>


using namespace cv;
using namespace std;


int MAX_KERNEL_LENGTH = 15;
int KERNEL_SIZE = 13;
float alpha = 0.4; //ค่ากำหนดความโปร่งแสงของสี
RNG rng(12345);
Mat morpho_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat morpho_kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
Tracker tracker = Tracker();

int main(int argc, char** argv)
{

	Mat capt, acc, background, motion, frame;

	//open video
	VideoCapture vid("video.avi");

	//check if we succeeded
	if (!vid.isOpened()) {
		return -1;
	}
	vid.read(capt);

	//แปลงเป็นภาพขาวดำ
	acc = Mat::zeros(capt.size(), CV_32FC3);

	int amount_up = 0;
	int amount_down = 0;

	for (;;) {
		vid >> frame;
		if (frame.empty()) 			
			break;

		imshow("motion detected", motion);		
		
		//blur video (Low-pass filter)
		GaussianBlur(frame, capt, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0);

		//delete background
		accumulateWeighted(capt, acc, alpha);
		convertScaleAbs(acc, background);
		subtract(capt, background, motion);

		//แปลงภาพ RGB เป็น gray scale แปลงเป็นภาพ binary
		imshow("motion threshold", capt);
		threshold(motion, capt, 20, 255, THRESH_BINARY);
		cvtColor(capt, capt, COLOR_BGR2GRAY);
		threshold(capt, capt, 5, 255, THRESH_BINARY);

		morphologyEx(capt, capt, MORPH_OPEN, morpho_kernel);
		dilate(capt, capt, morpho_kernel, Point(3, 3), 2);

		//Contour
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(capt, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		//Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());
		vector<Rect> classifiedRect;

		for (int i = 0; i < contours.size(); i++)
		{

			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			if ((boundRect[i].height > 60 && boundRect[i].width > 80) && (boundRect[i].height < 200 && boundRect[i].width < 200))
				classifiedRect.push_back(boundRect[i]);
			minEnclosingCircle(contours_poly[i], center[i], radius[i]);
		}

		Mat drawing = frame;

		//เส้นอ้างอิงในการนับจำนวนคน
		line(drawing, Point(0, drawing.rows / 3), Point(frame.cols, frame.rows / 3), Scalar(0, 255, 255), 2);

		// Draw polygonal contour + bonding rects + circles

		for (int i = 0; i < classifiedRect.size(); i++)
		{
			Scalar color = tracker.Track(classifiedRect[i], drawing.rows);
			rectangle(drawing, classifiedRect[i].tl(), classifiedRect[i].br(), color, 2, 8, 0);
			
			int begin = classifiedRect[i].y;
			int end = begin + classifiedRect[i].height;
			if ((begin < (drawing.rows / 3)) && (end > (drawing.rows / 3))) {
				int up_or_down = tracker.rectCounter(classifiedRect[i]);
				if (up_or_down == 1) {
					amount_up++;
				}
				else if (up_or_down == -1) {
					amount_down++;
				}
			}
		}
		cout << tracker.objsSet->objs.size() << endl;

		//set การแสดงผล output ของการนับ
		imshow("output", drawing);
		putText(drawing, "Amount up: ", Point(400, 400), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
		putText(drawing, to_string(amount_up), Point(570, 400), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 2, 2);
		putText(drawing, "Amount down: ", Point(400, 430), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 2, 2);
		putText(drawing, to_string(amount_down), Point(600, 430), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 2, 2);

		
		if (waitKey(10) >= 0) break;
	}

	return 0;
}


