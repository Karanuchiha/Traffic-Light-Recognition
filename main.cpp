#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <math.h>
#pragma once
#include <opencv\highgui.h>
#include <opencv\cv.h>
#include<math.h>
#include <opencv2\opencv.hpp>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

Mat image; Mat gray; Mat thresholded; Mat img_hsv; Mat hue; Mat final;

//Struct storing each object detected as a traffic light by the program, containing the bounding rectangle and the state if recognised
struct trafficLight {
public:
	Rect bounding_box;
	char* state = NULL;
};

//Class for Watershed Segmentation
class WatershedSegmenter {
private:
	Mat markers;

public:
	void setMarkers(const Mat& markerImage) {
		markerImage.convertTo(markers, CV_32S);
	}

	Mat process(Mat& image) {
		watershed(image, markers);
		return markers;
	}

	Mat getSegmentation() {
		Mat tmp;
		markers.convertTo(tmp, CV_8U);
		return tmp;
	}


};

bool isOverlapping(Rect my_rect, Rect true_rect) {
	double intersection_area = (my_rect&true_rect).area();
	if (intersection_area > 0)
		return true;
	else
		return false;
}

//Takes two rectangles and checks constraints to determine if the object was correctly located
bool isCorrectlyLocated(Rect my_rect, Rect true_rect) {
	double my_area = my_rect.area();
	double true_area = true_rect.area();
	double intersection_area = (my_rect&true_rect).area();
	double area_diff = abs(true_area - my_area);

	//Overlap more than 80% and difference in area within 20%
	if ((intersection_area / true_area >= 0.8) && (area_diff / true_area <= 0.2))
		return true;
	else return false;
}


//Calculating Dice coefficient for detected and truth rectangle
double getDiceCoeff(Rect my_rect, Rect true_rect) {
	double my_area = my_rect.area();
	double true_area = true_rect.area();
	double intersection_area = (my_rect&true_rect).area();

	double dice_coeff = (2 * intersection_area) / (my_area + true_area);
	return dice_coeff;
}

//Function to create a mask by thresholding the region of interest according to the minimum and maximum hue and saturation values
void getHSmask(const Mat& roi, double minHue, double maxHue, double minSat, double maxSat, Mat& mask) {
	Mat hsv;
	cvtColor(roi, hsv, CV_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);

	Mat mask1;
	threshold(channels[0], mask1, maxHue, 255, THRESH_BINARY_INV);
	Mat mask2;
	threshold(channels[0], mask2, minHue, 255, THRESH_BINARY);

	Mat hueMask;
	if (minHue < maxHue)
		hueMask = mask1 & mask2;
	else
		hueMask = mask1 | mask2;
	Mat satMask;
	inRange(channels[1], minSat, maxSat, satMask);
	mask = hueMask & satMask;
}

//Function that processes an ROI by appylying thresholding and closing(dilation followed by erosion)
Mat processROI(String color, Mat roi) {
	Mat mask;
	if(color == "green"){
		getHSmask(roi, 70, 95, 100, 255, mask);
	}
	else if (color == "red") {
		getHSmask(roi, 170, 5, 100, 255, mask);
	}
	else if(color == "amber")
		getHSmask(roi, 6, 50, 100, 255, mask);
	
	Mat detected(roi.size(), CV_8UC3, Scalar(0, 0, 0));
	//Copy roi to new Mat variable and apply Hue-Saturation mask
	roi.copyTo(detected, mask);
	cvtColor(detected, detected, CV_BGR2GRAY);
	threshold(detected, detected, 30, 255, THRESH_BINARY);

	dilate(detected, detected, Mat(), Point(-1, -1), 1);
	erode(detected, detected, Mat(), Point(-1, -1), 1);

	return detected;
}

//Finds contours in the processed ROI and draws the contour with the specified color in the final image
bool findLights(Mat detected, double roi_area, Rect roi_rect, Mat &final, String color) {
	vector<vector<Point>> contours1;
	vector<Vec4i> hierarchy1;
	bool isLight = false;
	Scalar lightColor;
	if (color == "green")
		lightColor = Scalar(0, 255, 0);
	if (color == "red")
		lightColor = Scalar(0, 0, 255);
	else if(color == "amber")
		lightColor = Scalar(0, 215, 255);

	findContours(detected, contours1, hierarchy1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (int j = 0; j < contours1.size(); j++) {
		
		if (contourArea(contours1[j]) < roi_area / 2 && contourArea(contours1[j]) > 30) {
			drawContours(final, contours1, j, lightColor, 2, CV_AA, hierarchy1, 2, Point(roi_rect.x, roi_rect.y));
			//Rect bound_rect = boundingRect(contours1[j]);
			//rectangle(contour_rois[i], bound_rect, Scalar(0, 0, 255), 2);
			isLight = true;			
		}
	}
	
	return isLight;
}


// Main image processing function
vector<struct trafficLight> processImage(String imageName, int idx) {
	
	vector<struct trafficLight> detected_lights;
	
	image = imread("C:\\Users\\Srijan\\Desktop\\Course\\Computer Vision\\OpenCVExample\\Media\\" + imageName);
	//Convert to gray-scale
	cvtColor(image, gray, CV_BGR2GRAY);
	//Apply threshold
	threshold(gray, thresholded, 35, 255, cv::THRESH_BINARY_INV);
	//Open
	erode(thresholded, thresholded, Mat(), Point(-1, -1), 1);
	dilate(thresholded, thresholded, Mat(), Point(-1, -1), 1);
	//Close
	dilate(thresholded, thresholded, Mat(), Point(-1, -1), 1);
	erode(thresholded, thresholded, Mat(), Point(-1, -1), 1);

	//Find foreground and background for watershed markers
	Mat fg;
	erode(thresholded, fg, Mat(), Point(-1, -1), 5);
	Mat bg;
	dilate(thresholded, bg, Mat(), Point(-1, -1), 5);
	threshold(bg, bg, 1, 128, THRESH_BINARY_INV);

	//Create marker image
	Mat markers(thresholded.size(), CV_8U, Scalar(0));
	markers = fg + bg;
	
	//Watershed segmentation
	WatershedSegmenter segmenter;
	segmenter.setMarkers(markers);
	segmenter.process(image);
	Mat segmented = segmenter.getSegmentation();


	threshold(segmented, segmented, 180, 255, THRESH_BINARY);
	
	//Cloning original image
	final = image.clone();

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Mat> contour_rois;
	vector<Rect> bounding_rects;
	
	//Find all contours
	findContours(segmented, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++) {
		//Calculate contour area
		double area = contourArea(contours[i]) + contours[i].size() / 2 + 1;
		for (int hole_number = hierarchy[i][2]; (hole_number >= 0); hole_number = hierarchy[hole_number][0])
			area = area - (contourArea(contours[hole_number]) - (contours[hole_number].size() / 2) + 1);

		//Get minAreaRect and corresponding area
		RotatedRect min_rect = minAreaRect(contours[i]);
		Size2f min_rect_size = min_rect.size;
		double rect_area = min_rect_size.area();


		Point2f* vertices;
		vertices = new Point2f[4];
		//Apply filters to check rectangularity and area
		if ((area / rect_area) > 0.87 && ((2 * min_rect_size.width < min_rect_size.height) || (2 * min_rect_size.height < min_rect_size.width)) && area > 600) {

			//Extract region of interest
			Mat roi = Mat(image, min_rect.boundingRect());
			contour_rois.push_back(roi);
			bounding_rects.push_back(min_rect.boundingRect());

			//Draw Rectangle in final image
			min_rect.points(vertices);
			for (int j = 0; j < 4; j++) {
				line(final, vertices[j], vertices[(j + 1) % 4], Scalar(255, 0, 0), 2, LINE_AA, 0);
			}
		}

		delete(vertices);
	}

	
	//Find Lights in ROIs
	for (int i = 0; i < contour_rois.size(); i++) {
		double roi_area = contour_rois[i].rows*contour_rois[i].cols;
		int num_lights = 0;
		bool isGreen = false;
		bool isRed = false;
		bool isAmber = false;
		struct trafficLight currLight;
		currLight.bounding_box = bounding_rects[i];

		Mat detected = processROI("red", contour_rois[i]);
		isRed = findLights(detected, roi_area, bounding_rects[i], final, "red");

		detected = processROI("amber", contour_rois[i]);
		isAmber = findLights(detected, roi_area, bounding_rects[i], final, "amber");

		if (isRed&&isAmber) {
			currLight.state = "Red+Amber";
		}

		else if (isRed) {
			currLight.state = "Red";
		}
		else if (isAmber)
			currLight.state = "Amber";

		detected = processROI("green", contour_rois[i]);
		isGreen = findLights(detected, roi_area, bounding_rects[i], final, "green");
		//imshow("light" + i, detected);
		if (isGreen) {
			currLight.state = "Green";
		}

		detected_lights.push_back(currLight);
	}

	imshow(to_string(idx), final);
	return detected_lights;

}


void getPerformanceMetrics(int i, int truth_boxes[], char* states[], vector<struct trafficLight> detected_lights, int &tp, int &fn, int& fp, double& dice_coeff, int& correct_state) {
	
	int num_traffic_lights = 0;
	//For images before the 12th image
	if (i < 11) {
		//Get truth rectangles
		Rect true_rect_1 = Rect(truth_boxes[8 * i], truth_boxes[8 * i + 1], truth_boxes[8 * i + 2] - truth_boxes[8 * i],
			truth_boxes[8 * i + 3] - truth_boxes[8 * i + 1]);
		Rect true_rect_2 = Rect(truth_boxes[8 * i + 4], truth_boxes[8 * i + 5],
			truth_boxes[8 * i + 6] - truth_boxes[8 * i + 4], truth_boxes[8 * i + 7] - truth_boxes[8 * i + 5]);
		
		//Compare detected rects with truth rects
		for (int j = 0; j < detected_lights.size(); j++) {
			char* my_state = detected_lights[j].state;
			Rect my_rect = detected_lights[j].bounding_box;

			//If it is overlapping with any of the truth rects, is it not a false negative but may be a false positive
			if (isOverlapping(my_rect, true_rect_1) || isOverlapping(my_rect, true_rect_2))
				num_traffic_lights++;

			if (isCorrectlyLocated(my_rect, true_rect_1)) {
				++tp;				
				dice_coeff += getDiceCoeff(my_rect, true_rect_1);
				if (my_state != NULL && (my_state == states[2 * i]))
					correct_state++;
				break;
			}
			else if (isCorrectlyLocated(my_rect, true_rect_2)) {
				++tp;				
				dice_coeff += getDiceCoeff(my_rect, true_rect_2);

				if (my_state != NULL && (my_state == states[2 * i + 1]))
					correct_state++;
				break;
			}
			else {
				++fp;
			}

		}
		fn += 2 - num_traffic_lights;
	}

	//For 12th image as it contains 4 lights
	else if (i == 11) {
		Rect true_rect_1 = Rect(truth_boxes[8 * i], truth_boxes[8 * i + 1], truth_boxes[8 * i + 2] - truth_boxes[8 * i],
			truth_boxes[8 * i + 3] - truth_boxes[8 * i + 1]);
		Rect true_rect_2 = Rect(truth_boxes[8 * i + 4], truth_boxes[8 * i + 5],
			truth_boxes[8 * i + 6] - truth_boxes[8 * i + 4], truth_boxes[8 * i + 7] - truth_boxes[8 * i + 5]);
		Rect true_rect_3 = Rect(truth_boxes[8 * i + 8], truth_boxes[8 * i + 9],
			truth_boxes[8 * i + 10] - truth_boxes[8 * i + 8], truth_boxes[8 * i + 11] - truth_boxes[8 * i + 9]);
		Rect true_rect_4 = Rect(truth_boxes[8 * i + 12], truth_boxes[8 * i + 13],
			truth_boxes[8 * i + 14] - truth_boxes[8 * i + 12], truth_boxes[8 * i + 15] - truth_boxes[8 * i + 13]);

		for (int j = 0; j < detected_lights.size(); j++) {
			char* my_state = detected_lights[j].state;
			Rect my_rect = detected_lights[j].bounding_box;
			if (isOverlapping(my_rect, true_rect_1) || isOverlapping(my_rect, true_rect_2) || isOverlapping(my_rect, true_rect_3) || isOverlapping(my_rect, true_rect_4))
				num_traffic_lights++;

			if (isCorrectlyLocated(my_rect, true_rect_1)) {
				++tp;
				++num_traffic_lights;
				dice_coeff += getDiceCoeff(my_rect, true_rect_1);
				if (my_state != NULL && (my_state == states[2 * i]))
					correct_state++;
				break;
			}
			else if (isCorrectlyLocated(my_rect, true_rect_2)) {
				++tp;
				++num_traffic_lights;
				dice_coeff += getDiceCoeff(my_rect, true_rect_2);
				if (my_state != NULL && (my_state == states[2 * i + 1]))
					correct_state++;
				break;
			}
			else if (isCorrectlyLocated(my_rect, true_rect_3)) {
				++tp;
				++num_traffic_lights;
				dice_coeff += getDiceCoeff(my_rect, true_rect_3);

				if (my_state != NULL && (my_state == states[2 * i + 2]))
					correct_state++;
				break;
			}
			else if (isCorrectlyLocated(my_rect, true_rect_4)) {
				++tp;
				++num_traffic_lights;
				dice_coeff += getDiceCoeff(my_rect, true_rect_4);

				if (my_state != NULL && (my_state == states[2 * i + 3]))
					correct_state++;
				break;
			}
			else {
				++fp;
			}
		}
		fn += 4 - num_traffic_lights;
	}
	//For images after the 12th image
	else {
		Rect true_rect_1 = Rect(truth_boxes[8 * i + 8], truth_boxes[8 * i + 9], truth_boxes[8 * i + 10] - truth_boxes[8 * i + 8],
			truth_boxes[8 * i + 11] - truth_boxes[8 * i + 9]);
		Rect true_rect_2 = Rect(truth_boxes[8 * i + 12], truth_boxes[8 * i + 13],
			truth_boxes[8 * i + 14] - truth_boxes[8 * i + 12], truth_boxes[8 * i + 15] - truth_boxes[8 * i + 13]);

		for (int j = 0; j < detected_lights.size(); j++) {
			char* my_state = detected_lights[j].state;
			Rect my_rect = detected_lights[j].bounding_box;

			if (isOverlapping(my_rect, true_rect_1) || isOverlapping(my_rect, true_rect_2))
				num_traffic_lights++;

			if (isCorrectlyLocated(my_rect, true_rect_1)) {
				++tp;
				++num_traffic_lights;
				dice_coeff += getDiceCoeff(my_rect, true_rect_1);
				if (my_state != NULL && (my_state == states[2 * i + 2]))
					correct_state++;
				break;
			}
			else if (isCorrectlyLocated(my_rect, true_rect_2)) {
				++tp;
				++num_traffic_lights;
				dice_coeff += getDiceCoeff(my_rect, true_rect_2);

				if (my_state != NULL && (my_state == states[2 * i + 3]))
					correct_state++;
				break;
			}
			else {
				++fp;
			}

		}
		fn += 2 - num_traffic_lights;
	}

}

int main(int argc, char** argv)
{
	char* images[] = {
		"CamVidLights01.png",
		"CamVidLights02.png",
		"CamVidLights03.png",
		"CamVidLights04.png",
		"CamVidLights05.png",
		"CamVidLights06.png",
		"CamVidLights07.png",
		"CamVidLights08.png",
		"CamVidLights09.png",
		"CamVidLights10.png",
		"CamVidLights11.png",
		"CamVidLights12.png",
		"CamVidLights13.png",
		"CamVidLights14.png",
	};

	int truth_boxes[] = { 
		319,202,346,279,
		692,264,711,322,
		217,103,261,230,
		794,212,820,294,
		347,210,373,287,
		701,259,720,318,
		271,65,309,189,
		640,260,652,301,
		261,61,302,193,
		644,269,657,312,
		238,42,284,187,
		650,279,663,323,
		307,231,328,297,
		747,266,764,321,
		280,216,305,296,
		795,253,816,316,
		359,246,380,305,
		630,279,646,327,
		260,122,299,239,
		691,271,705,315,
		331,260,349,312,
		663,280,676,322,
		373,219,394,279,
		715,242,732,299,
		423,316,429,329,
		516,312,521,328,
		283,211,299,261,
		604,233,620,279,
		294,188,315,253,
		719,225,740,286,
	};
	char* states[] = { "Green","Green","Green","Green","Green","Green","Red","Red","Red+Amber","Red+Amber","Green","Green","Amber","Amber","Amber","Amber","Green"
		,"Green","Green","Green","Green","Green","Green","Green","Red","Red","Red","Red","Red","Red" };

	int fp = 0;
	int fn = 0;
	int tn = 0;
	int tp = 0;
	int correct_state = 0;
	double dice_coeff = 0;

	double precision = 0, accuracy = 0, recall = 0;

	//Iterate through all images, process the images and get performance indicators
	for (int i = 0; i < 14; i++) {
		vector<struct trafficLight> detected_lights = processImage(images[i], i);
		getPerformanceMetrics(i, truth_boxes, states, detected_lights,
			tp, fn, fp, dice_coeff, correct_state);
	}

	//Compute metrics
	recall = (double)tp / ((double)tp + (double)fn);
	precision = (double)tp / ((double)tp + (double)fp);
	accuracy = (double)tp / 30;
	dice_coeff = dice_coeff / (double)30;

	//Display Metrics
	cout << "Accuracy: " << accuracy << endl;
	cout << "Precision: " << precision << endl;
	cout << "Recall:" << recall << endl;
	cout << "Percentage of lights with correct state: " << ((float)correct_state / 30) * 100 << "%"<<endl; 
	cout << "Average Dice Coefficient: " << dice_coeff;
	waitKey(0);

	return 0;
	   
}