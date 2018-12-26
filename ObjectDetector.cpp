#include "ObjectDetector.h"
using namespace cv;
using namespace std;


ObjectDetector::ObjectDetector()
{
}


ObjectDetector::~ObjectDetector()
{
}

void ObjectDetector::loaddetectFile(std::string fname)
{
	detector.load(fname);
}

void ObjectDetector::verifySVM()
{
	string verify = "vef";
	cout << "\nVerification data folder : ";
	cin >> verify;
	cout << "\nRunning matrix calculation on : " << verify << " folder";
	vector<String> files;
	glob(verify + "\\*.png", files);
	int pos = 0, neg = 0;
	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat im = imread(files[i]);
		resize(im, im, detector.winSize);
		Mat res;
		vector<float>descp;
		vector<Mat> in;
		detector.compute(im, descp);
		Mat inp = Mat(descp);
		in.push_back(inp);
		Mat dsc = m_convertToMat(in);
		svmapi->predict(dsc, res);
		float pb = res.at<float>(0, 0);
		//cout << "\nRes : " << pb;
		if (pb > 0.50)
			++pos;
		else
			++neg;
		//imshow("Plate", im);
		//if(waitKey(1) == 'q')
		//	break;
	}
	cout << "\nAccuracy matrix : ";
	float accper = (pos /(float)files.size());
	cout << "\nPositive : " << accper <<" %";
	cout << "\nNegative : " << 1 - accper;
}

void ObjectDetector::setsize(int h, int w)
{
	objSize = cv::Size(w, h);
}

void ObjectDetector::train()
{
	double c = 0.2,gamma = 0.1;
	int winH = 0, winW = 0;
	std::cout << "\nEnter SVM C : ";
	cin >> c;
	std::cout << "\nSVM gamma : ";
	cin >> gamma;
	
	std::string pd = "pd", nd = "nd";
	std::vector<cv::String> pvfiles,nvfiles;
	std::vector<cv::Mat> pvimgs,nvimgs,descp;
	cv::glob(pd + "\\*.png", pvfiles);
	cv::glob(nd + "\\*.png", nvfiles);
	std::cout << "\nPositive samples : " << pvfiles.size() << "\nNegative Samples : " << nvfiles.size();
	
	std::cout << "\nReading +ve images!";
	for (size_t i = 0; i < pvfiles.size(); ++i)
	{
		cv::Mat im = cv::imread(pvfiles[i]);
		cv::resize(im, im, objSize);
		pvimgs.push_back(im);
	}
	computeHog(pvimgs, descp);
	int pvsamp =(int)pvimgs.size();
	std::cout << "\nReading -ve images!";
	for (size_t j = 0; j < nvfiles.size(); ++j)
	{
		cv::Mat im = cv::imread(nvfiles[j]);
		cv::resize(im, im, objSize);
		nvimgs.push_back(im);
	}
	computeHog(nvimgs, descp);
	cv::Mat td = m_convertToMat(descp);
	std::cout << "\nData Size : " << td.size();
	std::vector<int> trainLabels;
	trainLabels.assign(pvsamp, +1);
	trainLabels.insert(trainLabels.end(), nvimgs.size(), -1);
	svmapi = cv::ml::SVM::create();
	svmapi->setCoef0(0.0);
	svmapi->setDegree(3);
	svmapi->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 3000, 1e-3));
	svmapi->setGamma(gamma);
	svmapi->setKernel(cv::ml::SVM::LINEAR);
	svmapi->setNu(0.5);
	svmapi->setP(5e-3);
	svmapi->setC(c); 
	svmapi->setType(cv::ml::SVM::EPS_SVR);
	std::cout << "\nTraining SVM!";

	svmapi->train(td, cv::ml::ROW_SAMPLE, trainLabels);
	svmapi->save("SVM.xml");
	std::cout << "\nDone Training!";
	verifySVM();
	std::vector<float> svmHogDescriptor = getSVMDescriptors(svmapi);
	detector.setSVMDetector(svmHogDescriptor);
	detector.save("LPR.xml");

}

vector<float> ObjectDetector::getSVMDescriptors(cv::Ptr<cv::ml::SVM>& svm)
{
	cv::Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	cv::Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);

	std::vector< float > hog_detector(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
	return hog_detector;
}

void ObjectDetector::detectobj(cv::Mat _inpimg,std::vector<cv::Rect>& objbbox, std::vector<double>& prob, float scale)
{
	detector.detectMultiScale(_inpimg, objbbox, prob , 0.0, Size(), Size(), scale);
}

cv::Mat ObjectDetector::m_convertToMat(std::vector<cv::Mat> hogimg)
{
	const int rows = (int)hogimg.size();
	const int cols = (int)std::max(hogimg[0].cols, hogimg[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	cv::Mat trainData = cv::Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < hogimg.size(); ++i)
	{
		CV_Assert(hogimg[i].cols == 1 || hogimg[i].rows == 1);
		if (hogimg[i].cols == 1)
		{
			cv::transpose(hogimg[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (hogimg[i].rows == 1)
		{
			hogimg[i].copyTo(trainData.row((int)i));
		}
	}
	return trainData;
}

int ObjectDetector::computeHog(std::vector<cv::Mat> images, std::vector<cv::Mat>& hogDescriptor)
{
	cv::Size cellsize(8, 8);
	cv::Size mblockSize = cellsize * 2;
	cv::Size mblockStride = cellsize;
	detector.cellSize = cellsize;
	detector.blockSize = mblockSize;
	detector.blockStride = mblockStride;
	detector.nbins = 11;
	detector.gammaCorrection = true;
	detector.winSize = objSize;
	
	std::vector<float> hogDescrib;
	for (int i = 0; i < (int)images.size(); ++i)
	{
		hogDescrib.clear();
		cv::Mat im = images[i];
		if (im.channels() > 2)
			cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
		cv::resize(im, im, objSize, cv::INTER_LANCZOS4);
		detector.compute(im, hogDescrib);
		if (i == 0)
		{
			std::cout << "\nDescriptor Size : " << hogDescrib.size() << "\n";
		}
		float per = (i / (float)images.size()) * 100;
		std::cout << "\r " << "Compute hog : \t [ " << std::roundf(per) << " % ] Completed!";
		hogDescriptor.push_back(cv::Mat(hogDescrib).clone());
	}
	return 0;
}

