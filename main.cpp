#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "opencv2/core/version.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "yolo_v2_class.hpp"	        // imported functions from .so

using namespace cv;
using namespace std;

void draw_boxes(cv::Mat& src, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, unsigned int wait_msec = 0) {
	for (auto &i : result_vec) {

        Rect rec(i.x, i.y, i.w, i.h);
        rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);
        putText(src, format("%s", obj_names[i.obj_id].c_str()), Point(i.x, i.y-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 1, 8, 0);

//		cv::Scalar color(60, 160, 260);
//		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
//		if(obj_names.size() > i.obj_id)
//			putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
//		if(i.track_id > 0)
//			putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x+5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
	}
}


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	for (auto &i : result_vec) {
		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		std::cout << std::setprecision(3) << "prob = " << i.prob << ",  x = " << i.x << ", y = " << i.y
			<< ", w = " << i.w << ", h = " << i.h << std::endl;
	}
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for(std::string line; file >> line;) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

int main()
{
    float f;
    float FPS[16];
    int i, Fcnt=0;
    Mat frame;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

	Detector Net("yolov4-tiny.cfg", "yolov4-tiny.weights");
	auto Names = objects_names_from_file("coco.names");

    VideoCapture cap("James.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
	while(1){
		try {
//		    frame = imread("parking.jpg");
            cap >> frame;
            if (frame.empty()) {
                cerr << "ERROR: Unable to grab from the camera" << endl;
                break;
            }

            Tbegin = chrono::steady_clock::now();

            //detect the objects
            vector<bbox_t> result = Net.detect(frame,0.35);

            Tend = chrono::steady_clock::now();

            //show the found objects
            draw_boxes(frame, result, Names);

            //calculate frame rate
            f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
            if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
            for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
            putText(frame, format("FPS %0.2f", f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));

            //show output
            imshow("Jetson Nano", frame);

            char esc = waitKey(5);
            if(esc == 27) break;

		}
		catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
		catch (...) { std::cerr << "unknown exception \n"; getchar(); }
	}

	return 0;
}
