#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define DISPLAY_IMAGE 1
// switch of different distance transform
// #define USE_L1_DISTANCE 

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

#if 0
// This is the sample code from: https://answers.opencv.org/question/129819/finding-distance-between-two-curves/
int main(int argc, const char** argv)
{
	cv::Mat image = cv::imread("14878460214049233.jpg", cv::IMREAD_GRAYSCALE);
	DisplayImage(image, "Source Image");

	cv::Mat binaryImage;
	cv::threshold(image, binaryImage, 200, 255, cv::THRESH_BINARY);
	std::cout << "Binary image type = " << cv::typeToString(binaryImage.type()) << std::endl;
	DisplayImage(binaryImage, "Binary Image");

	cv::Mat labels;
	cv::connectedComponents(binaryImage, labels);
	std::cout << "labels image type = " << cv::typeToString(labels.type()) << std::endl;
	cv::Mat result(image.size(), CV_32FC1, cv::Scalar::all(0));
    for (int i = 0; i <= 1; i++) {
        cv::Mat mask1 = labels == 1+i;
        cv::Mat mask2 = labels == 1+(1-i);
        cv::Mat masknot;
        bitwise_not(mask1,masknot);
		DisplayImage(mask1, "Mask1");
        cv::Mat dist;
        cv::distanceTransform(masknot, dist, cv::DIST_L2, 5, CV_8U);
		DisplayImage(dist / 255, "Distance Transform(distance float)");
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
#else
int main(int argc, const char** argv)
{
	cv::Mat image = cv::imread("NoDefect.bmp", cv::IMREAD_GRAYSCALE);
	DisplayImage(image, "Source Image");

	cv::Mat edgeImg;
	cv::Canny(image, edgeImg, 100, 200);
	DisplayImage(edgeImg, "Edge Image");
	cv::Mat edgeBinaryImage;
	cv::threshold(edgeImg, edgeBinaryImage, 100, 255, cv::THRESH_BINARY);
	DisplayImage(edgeBinaryImage, "Edge Binary Image");

	cv::Mat labels;
	cv::connectedComponents(edgeBinaryImage, labels);
	std::cout << "labels image type = " << cv::typeToString(labels.type()) << std::endl;
	cv::Mat result(image.size(), CV_32FC1, cv::Scalar::all(0));
    for (int i = 0; i <= 1; i++) {
        cv::Mat mask1 = labels == 1+i;
        cv::Mat mask2 = labels == 1+(1-i);
        cv::Mat masknot;
        bitwise_not(mask1,masknot);
		DisplayImage(mask1, "Mask1");
		DisplayImage(masknot, "masknot");
        cv::Mat dist;
#ifdef USE_L1_DISTANCE
        cv::distanceTransform(masknot, dist, cv::DIST_L1, 5, CV_8U);
		DisplayImage(dist, "Distance Transform(distance float)");
#else
        cv::distanceTransform(masknot, dist, cv::DIST_L2, 5, CV_8U);
		DisplayImage(dist / 255, "Distance Transform(distance float)");
#endif
		DisplayImage(mask2, "Mask2");
        dist.copyTo(result, mask2);
    }
	DisplayImage(result, "Distance");

	StoreMatToFile(result, "Image", "distCtr.yml");

    cv::SparseMat ms(result);
	StoreMatToFile(ms, "Mat", "sparseMat.yml");
#ifdef USE_L1_DISTANCE
    cv::SparseMatConstIterator_<uchar> it = ms.begin<uchar>(),it_end = ms.end<uchar>();
#else
    cv::SparseMatConstIterator_<float> it = ms.begin<float>(),it_end = ms.end<float>();
#endif
    cv::Mat lig(result.rows, 1, CV_8U, cv::Scalar::all(0));
    for (; it != it_end; it ++) {
        // print element indices and the element value
        const cv::SparseMat::Node* n = it.node();
        if (lig.at<uchar>(n->idx[0])==0) {
#ifdef USE_L1_DISTANCE
            std::cout << "(" << n->idx[0] << "," << n->idx[1] << ") = " << (int)it.value<uchar>() << "\t";
#else
            std::cout << "(" << n->idx[0] << "," << n->idx[1] << ") = " << it.value<float>() << "\t";
#endif
            lig.at<uchar>(n->idx[0])=1;
        }
    }

	// std::cout << "\n\nlig: \n" << lig << std::endl;

	return EXIT_SUCCESS;
}
#endif