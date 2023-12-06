#include <stdio.h>
#include <opencv2/opencv.hpp>

//using namespace cv;

long sml_af(cv::Mat im);

int main()
{
    std::string image_path = ".//images//cactus1.JPG";
    cv::Mat image, image_gray;
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);

    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    long part_1 = sml_af(image_gray(cv::Range(0, 100), cv::Range(0, 100)));
    long part_2 = sml_af(image_gray(cv::Range(500, 600), cv::Range(500, 600)));
    long part_3 = sml_af(image_gray(cv::Range(1000, 1100), cv::Range(1000, 1100)));
    // itd w ten sposob

    std::cout << "Part1 = " << part_1 << std::endl;
    std::cout << "Part2 = " << part_2 << std::endl;
    std::cout << "Part3 = " << part_3 << std::endl;

    return 0;
}

long sml_af(cv::Mat im)
{
    long sml = 0;

    for(int i = 1; i < (im.rows - 1); i++)
    {
        for(int j = 1; j < (im.cols - 1); j++)
        {
            sml += abs(4*im.at<char>(i, j) - im.at<char>(i-1, j) - im.at<char>(i, j-1) - im.at<char>(i, j+1) - im.at<char>(i+1, j));
        }
    }

    return sml;
}