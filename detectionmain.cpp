#include "ObjectDetector.h"


int main(int argc, char ** argv)
{
	ObjectDetector detector;
	if (argc < 2)
	{
		std::cout << "\nUsage : " << argv[0] << " <mode> <weight / Train xml > [scale] [ Img ]  ";
		return 0;
	}
	if (std::atoi(argv[1])==0)
	{
		int winH, winW;
		std::cout << "\nEnter object Height : ";
		cin >> winH;
		std::cout << "\nEnter object Widht : ";
		cin >> winW;
		detector.setsize(winH, winW);
		//detector.loaddetectFile(argv[2]);
		int64 ts = cv::getTickCount();
		detector.train();
		double tm = (cv::getTickCount() - ts) / cv::getTickFrequency() * 1000;
		std::cout << "\ntime : " << tm << " ms";

		return 0;
	}
	if (std::atoi(argv[1]) == 1)
	{
		detector.loaddetectFile(argv[2]);
		double scale = atof(argv[3]);
		if (scale <= 0)
			scale = 1.0;
		if (argc < 5)
		{
			char key = 'p';
			while (key != 'q')
			{
				std::cout << "\nEnetr image path : ";
				std::string imgname;
				std::cin >> imgname;
				cv::Mat im = cv::imread(imgname);
				if (!im.data)
					break;
				cv::resize(im, im, cv::Size(), scale, scale);
				std::vector<cv::Rect> loc;
				std::vector<double> weight;
				int64 ts = cv::getTickCount();
				detector.detectobj(im, loc, weight);
				double tm = (cv::getTickCount() - ts) / cv::getTickFrequency() * 1000;
				std::cout << "\ntime : " << tm << " ms";
				if (loc.size() == 0)
				{
					std::cout << "\nNo object detected!";
				}
				for (size_t i = 0; i < loc.size(); ++i)
				{
					stringstream ss;
					ss << i;
					cv::rectangle(im, loc[i], cv::Scalar(255, 0, 0));
					std::cout << "\nSize : " << loc[i].size();
					cv::putText(im, ss.str(), loc[i].tl(), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
					std::cout << "\nObject : " << i << " Prob : " << (float)(weight[i]*100)<< "%";
				}
				cv::imshow("Image", im);
				key = cv::waitKey();
			}
		}
		if (argc == 6)
		{
			cv::Mat im = cv::imread(argv[4]);
			std::vector<cv::Rect> loc;
			std::vector<double> weight;
			int64 ts = cv::getTickCount();
			detector.detectobj(im, loc, weight);
			double tm = (cv::getTickCount() - ts) / cv::getTickFrequency() * 1000;
			std::cout << "\ntime : " << tm << " ms";
			for (size_t i = 0; i < loc.size(); ++i)
			{
				stringstream ss;
				ss << i;
				cv::rectangle(im, loc[i], cv::Scalar(255, 0, 0));
				cv::putText(im, ss.str(), loc[i].tl(), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
				std::cout << "\nObject : " << i << " Prob : " << weight[i];
			}
			cv::imshow("Image", im);
			cv::waitKey();
		}
	}

	return 0;
}