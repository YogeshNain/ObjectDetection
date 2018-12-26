#include "ObjectDetector.h"
#include <Windows.h>

ObjectDetector detector;
typedef int(__stdcall *INITENGINE)(char *, char*, float, float);
typedef int(__stdcall *OCR)(unsigned char*, int, int, int, int, char *);

HMODULE dll;
INITENGINE LibInit;
OCR OCREngine;


cv::Rect deflateRect(cv::Rect _rt, float scale)
{
	cv::Rect bbox = _rt;
	bbox.x = _rt.x / scale;
	bbox.y = _rt.y / scale;
	bbox.height = _rt.height / scale;
	bbox.width = _rt.width / scale;
	return bbox;
}

int loadDetector(string fname)
{
	detector.loaddetectFile(fname);
	return 0;
}

int procMat(cv::Mat &im, float _resizeScale, float _thrValue)
{
	cv::Mat detectionimg;
	cv::resize(im, detectionimg, cv::Size(), _resizeScale, _resizeScale);
	std::vector<cv::Rect> loc;
	std::vector<double> weight;
	int64 ts = cv::getTickCount();
	detector.detectobj(detectionimg, loc, weight, _thrValue);
	double tm = (cv::getTickCount() - ts) / cv::getTickFrequency() * 1000;
	stringstream sstm;
	sstm << tm << " ms" << " s: " << _resizeScale << " t : " << _thrValue;
	cv::putText(im, sstm.str(), cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 20));
	//std::cout << "\ntime : " << tm << " ms";
	for (size_t i = 0; i < loc.size(); ++i)
	{
		stringstream ss, sp,bboxsp;
		ss << "ID : " <<i + 1 ;
		sp << cv::getTickCount();
		cv::Rect bbox = deflateRect(loc[i], _resizeScale);
		ss << "  Acc : " << weight[i];
		cv::Mat pltimg = cv::Mat(im, bbox);
		char ocrText[20];
		OCREngine(pltimg.data, pltimg.size().width, pltimg.size().height, 0, 0, ocrText);
		std::cout << "\nOCR : " << ocrText;
		bboxsp << bbox;
#ifdef NP
		if (weight[i] > 0.60)
			cv::imwrite("plt\\" + sp.str() + ".png", pltimg);
		else
			cv::imwrite("nplt\\" + sp.str() + ".png", pltimg);
#endif // NP
		cv::rectangle(im, bbox, cv::Scalar(255, 0, 0));
		//std::cout << "\nSize : " << loc[i].size();

		cv::putText(im, ss.str(), bbox.tl(), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
		cv::putText(im, bboxsp.str(), cv::Point(bbox.x,bbox.br().y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
		//std::cout << "\nObject : " << i + 1 << " Prob : " << (float)(weight[i] * 100) << "%";

	}
	if(im.size().width > 1300)
		cv::resize(im, im, cv::Size(1280, 720));
	cv::imshow("Image", im);
	return 0;
}

string parseCMDL(int argc, char **argv, char *key)
{
	string value;
	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], key) == 0)
		{
			if (i + 1 < argc)
			{
				//cout << "\n" << key << " : " << argv[i + 1];
				value = argv[i + 1];
			}
		}
	}
	return value;
}

int main(int argc, char ** argv)
{
	
	cv::String options =
		"\n{help h usage ?  |		|  print this message!"
		"Scale s	      |0.5	|  Scale for resize Image!"
		"Threshold t      |1.0>=|  Threshold to upscale and downscale"
		"Input i		  |input|  Image, Video or Image Folder <.png>"
		"Mode m			  |Mode |  Engine mode Train or Detect"
		"Weight w		  |Weight| Weight File of Weights";

	dll = LoadLibrary(L"LPREngine.dll");
	if (dll == NULL)
	{
		std::cout << "\n Can not find DLL";
	}
	LibInit = (INITENGINE)GetProcAddress(dll, "?InitEngine@CRxEngine@@QAGXPAD0MM@Z");
	if (LibInit == NULL)
	{
		std::cout << "\nCan not find Address!";
	}
	OCREngine = (OCR)GetProcAddress(dll, "?getOCR@CRxEngine@@QAGHPAEHHHHPAD@Z");

	if (OCREngine == NULL)
	{
		std::cout << "\nCan not find Address!";
	}

	if (argc < 2)
	{
		std::cout << "\nUsage : " << argv[0] << " -m mode -w weights -s scale -t thresh -i <Video or image/ folder>";
		std::cout << options;
		return 0;
	}
	

	//cv::CommandLineParser parser(argc, argv, options);

	float scale = 0.5;// parser.get<float>("s");
	scale = atof(parseCMDL(argc, argv, "-s").c_str());
	string modelweight = parseCMDL(argc, argv, "-w");//parser.get<string>("w");
	int mode = atoi(parseCMDL(argc, argv, "-m").c_str());// parser.get<int>("m");
	float threshScale = atof(parseCMDL(argc, argv, "-t").c_str());// parser.get<float>("t");
	string url = parseCMDL(argc, argv, "-i");//parser.get<string>("i");
	//cout << "\nMode : " << mode << "\nWeight : " << modelweight << "\nScale : " << scale ;
	loadDetector(modelweight);
	//return 0;
	if (mode == 0)
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

	LibInit("OCR\\SVM.xml", "OCR\\Hog.xml", 0.23, 0.66);
	if (mode == 1)
	{
		cv::namedWindow("Image", cv::WINDOW_FREERATIO);
		if (url.length() == 0 || (url.find(".jpg") != string::npos) || (url.find(".png") != string::npos)|| (url.find(".bmp") != string::npos))
		{
			char key = 'p';
			while (key != 'q')
			{

				std::string imgname;
				if (url.length() == 0)
				{
					std::cout << "\nEnetr image path : ";
					std::cin >> imgname;
				}
				else
					imgname = url;
				cv::Mat im = cv::imread(imgname);
				if (!im.data)
					break;
				procMat(im, scale, threshScale);
				key = cv::waitKey();
				url.clear();
			}
		}
		else
			if ((url.find(".mjp") != string::npos) || (url.find(".mp4") != string::npos) || (url.find(".avi") != string::npos))
			{
				cv::VideoCapture cap(url);
				if (!cap.isOpened())
				{
					std::cout << "\nCan not open Url : " << url;
					return 0;
				}

				std::cout << "\nOptions :\n p : Pause\n o : One By One \n - : scale-0.1 \n + : scale+0.1 \n u : thresh+0.1 \n d : thresh-0.1";
				std::cout << "\n r : Reset delay";
				char k = ' ';
				int delay = 27;
				while (1)
				{
					cv::Mat frm;
					cap.read(frm);
					if (!frm.data)
						break;
					procMat(frm, scale, threshScale);
					//cv::imshow("Image", frm);
					k = cv::waitKey(delay);
					if (k == 'p')
						k = cv::waitKey();
					if (k == 'q')
						break;
					if (k == 'o')
						delay = 0;
					if (k == '-')
						scale -= 0.1;
					if (k == '+')
						scale += 0.1;
					if (k == 'u')
						threshScale += 0.1;
					if (k == 'd')
						threshScale -= 0.1;
					if (k == 'r')
						delay = 27;
					frm.release();
				}

			}
			else
				if ((url.find(".png") == string::npos) || (url.find(".jpg")) == string::npos)
				{
					vector<cv::String> files;
					cv::glob(url + "\\*.png", files);
					cv::glob(url + "\\*.jpg", files);
					cv::glob(url + "\\*.bmp", files);
					for (size_t i = 0; i < files.size(); ++i)
					{
						cv::Mat im = cv::imread(files[i]);
						procMat(im, scale, threshScale);
						//cv::imshow("Image", im);
						char key = cv::waitKey();
						if (key == 'q')
							break;
					}
				}
				else
				{
					cout << "\nCan not open given url : " << url;
				}

	}

	return 0;
}