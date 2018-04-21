#pragma once

/*

Receipt.h

*** Contributions ***

Arjun Vegda (github: @arjunvegda) -

Extracted text using Tesseract OCR library
Bug fixes/ Optimization

Olga Belavina -
Recover Perspective


Rocco //LASTNAME
//TODO


*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <tesseract\baseapi.h>
#include <leptonica\allheaders.h>

#include <iostream>
#include <iostream>
#include <algorithm>    // std::sort
#include <vector>
using namespace cv;
using namespace std;


typedef std::vector<std::vector<cv::Point>> contourVector;
cv::RNG rng(12345);

const double maxnewWidth = 500;
const double maxnewHeight = 600;
const double scaleRatio = 0; //In percent

class Receipt {


	Mat img_;
	Mat grayScale_;
	Mat currentWorkingImg_;
	Mat finalOutput_;
	Mat edges_;
	contourVector receiptContours_;
	bool logProcess_;
	bool shouldRecoverPerspective_;
	string filename_;
	/*
	This function reads the image and stores it in img

	@param String filename - Takes in the path to the receipt file
	@param Mat& img - Reference to the image object. Defaulted to original image
	@param int flags - Allows to specify the flag for opencv's imread function defaulted to -1
	@return success status
	*/
	bool readImg(string filename, Mat& img, int flag = -1) {
		img = cv::imread(filename, flag);
		if (!img.data) {
			cout << "Unable to open the file " << filename << ".";
			return false;
		}

		filename_ = filename;
		return true;
	}

	/**
	* Detects receipt in an image by
	* 1) applying canny edge detection
	* 2) finding contours
	* 3) selecting receipt contour
	* 4) selecting receipt contour's corners
	* 5) applying transformation with
	*/
	void recoverPerspective(bool showCorners = false) {

		if (this->logProcess_) {
			cout << "- Running RecoverPerspective()\n";
		}

		cout << "- Should Display recovered perspective image? (Y/N)\n";
		bool displayFinalImg = this->getAffirmation_();

		cv::Mat output, greyScale, contoursImg;
		contourVector contours;
		std::vector<cv::Vec4i> hierarchy;

		if (this->logProcess_) {
			cout << "- Applying blur and finding Edges on the image\n";
		}
		GaussianBlur(this->grayScale_, this->currentWorkingImg_, cv::Size(5, 5), 0);

		if (this->logProcess_) {
			cout << "- Running Canny edge detector\n";
		}
		Canny(this->currentWorkingImg_, this->currentWorkingImg_, 75, 200, 3);


		this->edges_ = this->currentWorkingImg_.clone();

		if (this->logProcess_) {
			cout << "- Finding Contours\n";
		}

		/// Find Contours ///
		findContours(this->currentWorkingImg_, // stores detected edges
			contours,
			hierarchy,
			RETR_TREE,
			CHAIN_APPROX_SIMPLE
		);

		cv::Mat drawing = cv::Mat::zeros(this->currentWorkingImg_.size(), CV_8UC3);
		this->edges_ = cv::Mat::zeros(this->currentWorkingImg_.size(), CV_8UC3);

		if (this->logProcess_) {
			cout << "- Sorting contours by area\n";
		}

		// Sort by area
		std::sort(contours.begin(), contours.end(),
			[](const std::vector<cv::Point> & a, const std::vector<cv::Point> & b) -> bool
		{
			return cv::contourArea(a) > contourArea(b);
		});

		if (this->logProcess_) {
			cout << "- Looping through found contours to find the receipt\n";
		}

		// Loop through found contours & find the receipt
		for (int i = 0; i< contours.size(); i++)
		{
			// get a curve length of the contour
			double length = arcLength(contours[i], true);

			// approximate a polygonal curve(s) with the specified precision.
			// ('fit' a polygon into contours)
			std::vector<cv::Point> approx;
			approxPolyDP(contours[i], approx, 0.02f * length, true);

			// If countour has 4 points (meaning it's a rectangle) -> found receipt
			if (approx.size() == 4) {
				this->receiptContours_.push_back(approx);
				break;
			}

		}

		if (this->logProcess_) {
			cout << "- Drawing contours on the image\n" << std::endl;
		}

		for (int i = 0; i< contours.size(); i++)
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::drawContours(this->edges_, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
		}

		auto imgSize = this->img_.size();

		if (cv::contourArea(receiptContours_[0]) < 0.4 * (imgSize.width*imgSize.height)) {
			this->img_.copyTo(this->currentWorkingImg_);
			std::cout << "The Receipt does not need to recover perspective." << std::endl;
			return;
		}


		/* -- Draw receipt countours -- */
		for (int i = 0; i< this->receiptContours_.size(); i++)
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::drawContours(drawing, this->receiptContours_, i, color, 2, 8, hierarchy, 0, cv::Point());
		}

		cv::Point2f srcPoints[4], dstPoints[4];

		// Source
		srcPoints[0] = this->receiptContours_[0][0];
		srcPoints[1] = this->receiptContours_[0][1];
		srcPoints[2] = this->receiptContours_[0][2];
		srcPoints[3] = this->receiptContours_[0][3];

		// Destination Rectangle
		dstPoints[0] = Point2f(imgSize.width, 0);     // top-right corner
		dstPoints[1] = Point2f(0, 0);                 // top-left  corner
		dstPoints[2] = Point2f(0, imgSize.height);    // bottom-left corner
		dstPoints[3] = Point2f(imgSize.width, imgSize.height); // bottom-right corner

		/////////////////////////////////////////////////////////
		// Check if the order of the (source) corners matches
		// order of destination points by getting distances between
		// each srcPoint & dstPoint and finding minimum
		/////////////////////////////////////////////////////////

		std::vector<int> indexMap;
		cv::Point2f srcPointsCopy[4];

		for (int i = 0; i < 4; i++) {

			std::vector<float> distances;

			// find distances
			for (int j = 0; j < 4; j++) {
				distances.push_back(sqrt(pow(dstPoints[j].x - srcPoints[i].x, 2) + pow(dstPoints[j].y - srcPoints[i].y, 2)));
			}

			// find minimum (closest corner)
			auto minDistance = std::min_element(distances.begin(), distances.end());
			int dstPointIndex = std::distance(distances.begin(), minDistance);

			if (dstPointIndex != i) {
				std::cout << "\nNeed to move corner <" << i << "> to <" << dstPointIndex << ">" << std::endl;
			}

			// save index (where source point is supposed to be)
			indexMap.push_back(dstPointIndex);
			srcPointsCopy[i] = srcPoints[i];
		}


		if (this->logProcess_) {
			cout << "- Reordering corners\n";
		}


		// Reorder Corners
		for (int i = 0; i < 4; i++) {
			srcPoints[indexMap[i]] = srcPointsCopy[i];
		}

		if (showCorners) {
			if (this->logProcess_) {
				cout << "- Drawing corners\n";
			}
			for (int i = 0; i < 4; i++) {
				circle(this->currentWorkingImg_, srcPoints[i], 10 * (i + 1), cv::Scalar(255, 0, 0));
			}

			for (int i = 0; i < 4; i++) {
				circle(this->currentWorkingImg_, dstPoints[i], 40, cv::Scalar(0, 255, 0));
			}
		}

		if (this->logProcess_) {
			cout << "- Applying perspective\n";
		}

		// Transform Receipt -> apply perspective
		cv::Mat warp_mat = getPerspectiveTransform(srcPoints, dstPoints);
		//Mat temp = this->currentWorkingImg_.clone();
		//this->currentWorkingImg_ = this->img_.clone();
		this->img_.copyTo(this->currentWorkingImg_);
		warpPerspective(this->img_, this->currentWorkingImg_, warp_mat, this->currentWorkingImg_.size());

		if (displayFinalImg) {
			showImg("Receipt Perspective", this->currentWorkingImg_);
			showImg("Receipt Perspective Edges", this->edges_);
		}
	}

	/*
	This function extracts text from the receipt using Tesseract OCR library.
	*/
	void extractText() {

		if (this->logProcess_) {
			cout << "- Converting image to gradient\n";
		}

		Mat gradient;
		Mat morphStructure = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		morphologyEx(this->grayScale_, gradient, MORPH_GRADIENT, morphStructure);

		if (this->logProcess_) {
			cout << "- Converting image to Binary using THRESH_OTSU\n";
		}

		cout << "- Should Display Gradient image? (Y/N)\n";
		bool displayGradImg = this->getAffirmation_();
		gradient.copyTo(this->currentWorkingImg_);

		if (displayGradImg) {
			showImg("Receipt Gradient", this->currentWorkingImg_);
		}

		Mat binary;
		threshold(gradient, binary, 0.0, 255.0, THRESH_OTSU);



		cout << "Should Display Binary image? (Y/N)\n";
		bool displayBinImg = this->getAffirmation_();
		binary.copyTo(this->currentWorkingImg_);

		if (displayBinImg) {
			showImg("Receipt binary", this->currentWorkingImg_);
		}

		Mat closed;
		Size closeKernel = Size(11, 1);
		morphStructure = getStructuringElement(MORPH_RECT, closeKernel);

		if (this->logProcess_) {
			cout << "- Running Morph. closed with kernel " << closeKernel << "\n";
		}

		morphologyEx(binary, closed, MORPH_CLOSE, morphStructure);

		cout << "Should Display Closed image? (Y/N)\n";
		bool displayCloseImg = this->getAffirmation_();
		closed.copyTo(this->currentWorkingImg_);

		if (displayCloseImg){
			showImg("Receipt closed", this->currentWorkingImg_);
		}

		Mat connected;
		Mat finalImg;
		this->img_.copyTo(finalImg);
		Rect rect;
		// Find contours 
		Mat mask = Mat::zeros(this->img_.size(), CV_8UC1);
		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;
		closed.copyTo(connected);

		if (this->logProcess_) {
			cout << "- Finding contours to detect combined areas.\n";
		}

		findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		if (this->logProcess_) {
			cout << "- Initializing tesseract OCR API with English language.\n";
		}

		tesseract::TessBaseAPI tess;
		tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

		// Filter contours 
		for (int i = 0; i >= 0; i = hierarchy[i][0]) {
			rect = boundingRect(contours[i]);
			// Ignore if the rect is too small     
			if ((rect.height < 14 || rect.width < 14)) {
				continue;
			}
			Mat maskROI(mask, rect);
			maskROI = Scalar(0, 0, 0);
			drawContours(mask, contours, i, Scalar(255, 255, 255), CV_FILLED);

			// Calculate ratio of non-zero pixels in the filled region     
			double r = (double)countNonZero(maskROI) / (rect.width*rect.height);

			// If the ration is bigger than 45% we assume it contains texts     
			if (r > 0.45) {
				tess.SetRectangle(rect.x, rect.y, rect.width, rect.height);
				rectangle(finalImg, rect, Scalar(0, 0, 255), 2);
			}
		}

		cout << "Should display image with possible detected text? (Y/N)\n";
		bool displayRectImage = this->getAffirmation_();
		finalImg.copyTo(this->currentWorkingImg_);

		if (displayRectImage) {
			showImg("Possible text", this->currentWorkingImg_);
		}

		Mat tempBin;

		//threshold(actualGray, tempBin, 200, 255.0, THRESH_OTSU);

		if (this->logProcess_) {
			cout << "- Applying Median blur on the image.\n";
		}

		medianBlur(tempBin, tempBin, 1);
		//tempBin = tempBin - Scalar(75, 75, 75); // reduce brightness

		if (this->logProcess_) {
			cout << "- Converting image to Binary using Adaptive Threshold Gaussian.\n";
		}

		adaptiveThreshold(this->grayScale_, tempBin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 115, 20);


		Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 1));

		if (this->logProcess_) {
			cout << "- Dilating the binary image.\n";
		}

		dilate(tempBin, tempBin, kernel, Point(-1, -1), 3);

		if (this->logProcess_) {
			cout << "- Running Morph. Close on binary image.\n";
		}

		morphologyEx(tempBin, tempBin, MORPH_CLOSE, kernel);


		/*namedWindow("Group 7 - Morph Open Image 7 x 7", CV_WINDOW_AUTOSIZE);
		imshow("Group 7 - Morph Open Image 7 x 7", tempbin);*/

		if (this->logProcess_) {
			cout << "- Sending the altered binary image to Tesseract.\n";
		}

		// Pass it to Tesseract API
		tess.SetImage((uchar*)tempBin.data, tempBin.cols, tempBin.rows, 1, tempBin.cols);

		// Get the text
		if (this->logProcess_) {
			cout << "- Extracting the text.\n";
		}
		Boxa* boxes = tess.GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL);
		if (boxes) {
			cout << "\n * * * * * * * *\n";
			double avg = 0;
			for (int i = 0; i < boxes->n; i++) {
				BOX* box = boxaGetBox(boxes, i, L_CLONE);
				tess.SetRectangle(box->x, box->y, box->w, box->h);
				char* ocrResult = tess.GetUTF8Text();

				int conf = tess.MeanTextConf();
				avg += conf;
				cout << "Surity: " << conf << ", Text: " << ocrResult;

				//TODO - Might wanna store it in a file. Perhaps, csv.
				// << endl;
				/*fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s",
				i, box->x, box->y, box->w, box->h, conf, ocrResult);*/
			}
			cout << "\nAverage Surity: " << avg / boxes->n << endl;
			cout << "\n * * * * * * * *\n";
		}
		else {
			cout << "Unable to parse the text from the image...\nCheck the way you set rectangles on the image\n";
		}
	}

	void showImg(string windowName, Mat& img, int windowFlag = 1, int waitKeyOpt = 0) {
		if (!img.data) {
			cout << "Unable to display the image as its empty.\n";
		}
		else {
			namedWindow(this->filename_ + ' ' + windowName, windowFlag);
			imshow(this->filename_ + ' ' + windowName, scaleDownImg(&img));
		}
	}

	bool getAffirmation_() {
		char input = 'x';
		cout << "> ";
		cin >> input;

		cin.ignore();
		cin.clear();
		if (input == 'Y' || input == 'y') {
			return true;
		}
		else {
			return false;
		}

	}

	Mat scaleDownImg(Mat* input) {
		Mat resized = Mat();
		const int ogWidth = this->img_.cols;
		const int ogHeight = this->img_.rows;
		const double ratio = MIN(maxnewWidth / ogWidth, maxnewHeight / ogHeight);
		const int newWidth = ogWidth * ratio;
		const int newHeight = ogHeight * ratio;
		resize(*input, resized, Size(newWidth, newHeight), 0, 0, CV_INTER_LINEAR);
		return resized;
	}

public:

	Receipt(string filename, int readflag = -1) {
		cout << "Welcome to the receipt detector!\n";
		cout << "This program is built by Arjun, Olga and Rocco\n\n";

		if (imread(filename, readflag)) {
			cout << "Would you like to log the process to the screen? (Y/N)\n";
			if (this->getAffirmation_()) {
				cout << "Great! Will log the process to the screen\n";
				this->logProcess_ = true;
			}
			else {
				cout << "No problem! Will NOT log the process to the screen\n";
				this->logProcess_ = false;
			}

			cout << "Would you like to recover the perspective of the image? (Y/N)\n";
			if (this->getAffirmation_()) {
				cout << "Great! Will recover the perspective\n";
				this->shouldRecoverPerspective_ = true;
			}
			else {
				cout << "No problem! Will work with the current image.\n";
				this->shouldRecoverPerspective_ = false;
			}

			if (this->logProcess_) {
				cout << "Converting image to grayscale\n";
			}
			cvtColor(this->img_, this->grayScale_, CV_BGR2GRAY);

			return;
		}

	}


	/*
	This function returns a copy of the original image
	*/
	Mat getOriginalImage() {
		//Might wanna check if its empty before cloning.
		if (this->img_.data) {
			return scaleDownImg(&this->img_.clone());
		}

		cout << "Could not get original image as it is empty.\n";
		return Mat();
	}

	/*
	This is a wrapper function for readImg. Reads into img_
	*/
	bool imread(string filename, int flag) {
		if (this->readImg(filename, this->img_, flag)) {
			if (this->logProcess_) {
				cout << "Resizing the image by " << scaleRatio << "%\n";
			}

			resize(this->img_, this->img_, Size(this->img_.cols * (1 + (scaleRatio / 100)), this->img_.rows * (1 + (scaleRatio / 100))), 0, 0, CV_INTER_LINEAR);
			return true;
		}
		return false;
	}



	void doTheMagic() {

		if (this->img_.data) {

			cout << "\n**************************\n";
			cout << "** Casting magic spells **\n";
			cout << "**************************\n\n";

			showImg("Original Image", this->img_);
			int currentProcess = 0;

			if (this->shouldRecoverPerspective_) {
				if (this->logProcess_) {
					cout << "Step 1. Recover Perspective\n";
				}
				this->recoverPerspective();

				if (this->logProcess_) {
					cout << "Recover Perspective ended.\n\n";
					cout << "Step 2. Extract Text\n\n";
				}
			}
			else {
				if (this->logProcess_) {
					cout << "Step 1. Extract Text\n\n";
				}
			}

			this->extractText();

			cout << "Text extraction process ended.\n\n";
			cout << "Press any key on the image to exit\n";
			waitKey(0);
			destroyAllWindows();
			cout << "\n\nThank you for trying Receipt detector! Have a good day!\n";
		}
		else {
			cout << "\nSorry, cannot cast magic spells on an empty image. Please re-read the image and try again\n";
		}

	}

};