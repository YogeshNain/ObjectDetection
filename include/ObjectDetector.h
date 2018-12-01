#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class ObjectDetector
{
public:
	ObjectDetector();
	~ObjectDetector();

	void train();
	vector<float> getSVMDescriptors(cv::Ptr<cv::ml::SVM>& svm);
	void detectobj(cv::Mat _inpimg, std::vector<cv::Rect>& objbbox, std::vector<double>& prob);
	cv::Mat m_convertToMat(std::vector<cv::Mat> hogimg);
	int computeHog(std::vector<cv::Mat> images, std::vector<cv::Mat>& hogDescriptor);
	void loaddetectFile(std::string fname);
	void setsize(int h, int w);
private:
	cv::HOGDescriptor detector;
	cv::Ptr<cv::ml::SVM> svmapi;
	void verifySVM();
	
	Size objSize;
};

