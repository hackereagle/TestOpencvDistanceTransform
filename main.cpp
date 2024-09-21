#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define DISPLAY_IMAGE 1
// switch of different distance transform

void DisplayImage(const cv::Mat& image, const std::string& title)
{
#if DISPLAY_IMAGE
	cv::namedWindow(title, cv::WINDOW_NORMAL);
	std::cout << "Already display \"" << title << "\". Please press any key to continue..." << std::endl;
	cv::imshow(title, image);
	cv::waitKey(0);
#endif //DISPLAY_IMAGE
}

template<typename T>
void StoreMatToFile(const T& mat, const std::string& matName, const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << matName << mat;
    fs.release();
}

bool IsMaskEmpty(const cv::Mat& mask)
{
    bool isMaskEmpty = false;

	if (cv::countNonZero(mask) == 0)
        isMaskEmpty = true;

    return isMaskEmpty;
}

void DrawAllLabels(cv::Mat labels, cv::Mat& labelsResultImage)
{
    if (labels.type() != CV_32S) {
        std::cerr << "The input image type is not CV_32S" << std::endl;
        return;
    }

    cv::RNG rng(12345);
    // cv::Mat labelResultImage = cv::Mat::zeros(labels.size(), CV_8UC3);
    labelsResultImage = cv::Mat::zeros(labels.size(), CV_8UC3);
    cv::Mat colorBackground;
    cv::Mat oneLabel;
    int label = 1; // label start from 1. zero is background
    do {
        oneLabel = labels == label;
        if (IsMaskEmpty(oneLabel)) 
            break;
        
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        colorBackground = cv::Mat(labels.size(), CV_8UC3, color);
        colorBackground.copyTo(labelsResultImage, oneLabel);
        label = label + 1;
    } while (!IsMaskEmpty(oneLabel));

    // DisplayImage(labelResultImage, "Label Result Image");
}

int main(int argc, const char** argv)
{
	cv::Mat image = cv::imread("NoDefect.bmp", cv::IMREAD_GRAYSCALE);
	DisplayImage(image, "Source Image");

	cv::Mat edgeImg;
	cv::Canny(image, edgeImg, 100, 200);
	DisplayImage(edgeImg, "Edge Image");
	cv::Mat edgeBinaryImage;
	cv::threshold(edgeImg, edgeBinaryImage, 100, 255, cv::THRESH_BINARY);
	// DisplayImage(edgeBinaryImage, "Edge Binary Image");

	cv::Mat labels;
	int num = cv::connectedComponents(edgeBinaryImage, labels);
    std::cout << "labels image type = " << cv::typeToString(labels.type()) << ", label number = " << num << std::endl;
    cv::Mat labelsResultImage;
    DrawAllLabels(labels, labelsResultImage);
    DisplayImage(labelsResultImage, "edges");
	std::cout << "labels image type = " << cv::typeToString(labels.type()) << std::endl;
	cv::Mat result(image.size(), CV_32FC1, cv::Scalar::all(0));
    for (int i = 0; i <= 1; i++) {
        cv::Mat mask1 = labels == 1+i;
        cv::Mat mask2 = labels == 1+(1-i);
        cv::Mat masknot;
        bitwise_not(mask1,masknot);
		DisplayImage(mask1, "Mask1");
		DisplayImage(masknot, "masknot");

        cv::Mat dist, distLabels;
        cv::distanceTransform(masknot, dist, distLabels, cv::DIST_L2, 5);
		DisplayImage(dist / 255, "Distance Transform(distance float)");
        cv::Mat distLabelsResultImage;
        DrawAllLabels(distLabels, distLabelsResultImage);
        DisplayImage(distLabelsResultImage, "Distance Labels");

		DisplayImage(mask2, "Mask2");
        dist.copyTo(result, mask2);
    }
	DisplayImage(result, "Distance");

	StoreMatToFile(result, "Image", "distCtr.yml");

    cv::SparseMat ms(result);
	StoreMatToFile(ms, "Mat", "sparseMat.yml");
    cv::SparseMatConstIterator_<float> it = ms.begin<float>(),it_end = ms.end<float>();
    cv::Mat lig(result.rows, 1, CV_8U, cv::Scalar::all(0));
    for (; it != it_end; it ++) {
        // print element indices and the element value
        const cv::SparseMat::Node* n = it.node();
        if (lig.at<uchar>(n->idx[0])==0) {
            std::cout << "(" << n->idx[0] << "," << n->idx[1] << ") = " << it.value<float>() << "\t";
            lig.at<uchar>(n->idx[0])=1;
        }
    }

	// std::cout << "\n\nlig: \n" << lig << std::endl;

	return EXIT_SUCCESS;
}