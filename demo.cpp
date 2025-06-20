/*
*Copyright (c) 2020-, Ningbo Institute of Materials Technology and Engineering, Chinese Academy of Sciences
*@file demo.cpp
*@brief Multi frame registration example
*@author ZhouXiang
*@date 2025/06/20
* @version 1.1
*/
#pragma comment(lib, "opencv_world3412.lib")
#pragma comment(lib, "onnxruntime.lib")
#pragma comment(lib, "onnxruntime_providers_cuda.lib")
#pragma comment(lib, "onnxruntime_providers_shared.lib")
#pragma comment(lib, "onnxruntime_providers_tensorrt.lib")
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cstdio>
#include "LightGlue.h"
#include <fstream>
#include <nlohmann/json.hpp>
////Before you start the program, you'll need to check that the frame rate of the video is consistent between the narrow and wide fields of view
using json = nlohmann::json;
/*
* @transform the Mat to Json,just suport single channel Mat 
* @param: const cv::Mat& mat **Single-channel Mat object
*/
json mat_to_json(const cv::Mat& mat) {
	if (mat.channels() != 1) {
		throw std::runtime_error("channel num more than one");
	}
	json j;
	j["rows"] = mat.rows;
	j["cols"] = mat.cols;
	j["type_id"] = "opencv-matrix";
	j["dt"] = "d";

	json data = json::array();
	for (int r = 0; r < mat.rows; r++) {
		for (int c = 0; c < mat.cols; c++) {
			switch (mat.depth()) {
			case CV_8U:  data.push_back(mat.at<uchar>(r, c)); break;
			case CV_8S:  data.push_back(mat.at<char>(r, c)); break;
			case CV_16U: data.push_back(mat.at<ushort>(r, c)); break;
			case CV_16S: data.push_back(mat.at<short>(r, c)); break;
			case CV_32S: data.push_back(mat.at<int>(r, c)); break;
			case CV_32F: data.push_back(mat.at<float>(r, c)); break;
			case CV_64F: data.push_back(mat.at<double>(r, c)); break;
			default: throw std::runtime_error("Unsupported data types");
			}
		}
		
	}
	j["data"] = data;
	return j;
}



/*
* @Save the Homography  matrix as a JSON file
* @param 1: string filename	**The name of the saved file
* @param 2: cv::Mat M		**Homography  matrix
*/
int WriteJsonFile(std::string filename, cv::Mat M) 
{

	json j = mat_to_json(M);
	std::ofstream o(filename.c_str());
	o << std::setw(4) << j << std::endl;
	return 1;
}


std::string remove_extension_c(const char* filename) {
	const char* last_slash = std::max(strrchr(filename, '/'), strrchr(filename, '\\'));
	const char* start = last_slash ? last_slash + 1 : filename;

	const char* last_dot = strrchr(start, '.');

	if (!last_dot || last_dot == start) {
		return start; // No suffix or hidden file
	}

	return std::string(start, last_dot - start);
}


int main(int argc, char** argv)
{

	std::string narrow_video_flie = "narrow_demo.mp4";
	std::string wide_video_flie = "wide_demo.mp4";

	//check frames num

	cv::VideoCapture video_narrow;
	video_narrow.open(narrow_video_flie);
	if (!video_narrow.isOpened()) {
		std::cerr << "Error: Could not open video: " << narrow_video_flie << std::endl;
		return -1;
	}
	int narrow_frames_count = int(video_narrow.get(CV_CAP_PROP_FRAME_COUNT));
	std::cout << "narrow_frames_count:" << narrow_frames_count << std::endl;


	cv::VideoCapture video_wide;
	video_wide.open(wide_video_flie);
	if (!video_wide.isOpened()) {
		std::cerr << "Error: Could not open video: " << wide_video_flie << std::endl;
		return -1;
	}
	int wide_frames_count = int(video_wide.get(CV_CAP_PROP_FRAME_COUNT));
	std::cout << "wide_frames_count:" << wide_frames_count << std::endl;
	int frame_count = 0;
	if (narrow_frames_count != wide_frames_count)
	{
		std::cout << "The frame count of the video is inconsistent" << std::endl;
		frame_count = narrow_frames_count < wide_frames_count ? narrow_frames_count : wide_frames_count;
	}
	//load the lightglue&superpoint onnx model
	LightGlue lightglue(0.05, 0.2, "superpoint_731x1024.onnx", "superpoint_lightglue_fused_fp16.onnx");

	int narrow_frame_height = (int)video_narrow.get(CV_CAP_PROP_FRAME_HEIGHT);
	int narrow_frame_width = (int)video_narrow.get(CV_CAP_PROP_FRAME_WIDTH);

	int wide_frame_height = (int)video_wide.get(CV_CAP_PROP_FRAME_HEIGHT);
	int wide_frame_width = (int)video_wide.get(CV_CAP_PROP_FRAME_WIDTH);

	std::vector<std::vector< cv::Point2f> >mkpts;
	cv::Rect2f roi_narrow(0, 0, narrow_frame_width, narrow_frame_height);
	cv::Rect2f roi_wide(0, 0, wide_frame_width, wide_frame_height);
	cv::Mat last_M;
	std::vector<cv::Point2f> last_edge_points;

	FixedSizeQueue<cv::Mat> warped_images(5);
	FixedSizeQueue<std::vector<cv::Point2f>> edge_points(5);


	cv::Mat narrow_img;
	cv::Mat wide_img;
	int output_count = 1;
	//begin the match step
	while (output_count<frame_count)
	{
		static int count = 1;
		static int inliner_sum = 0;
		static int outliner_sum = 0;
		static double avg_edge_point_distance = 0;
		static double distance_sum = 0;
		video_wide.read(wide_img);
		video_narrow.read(narrow_img);
		cv::Mat wide_img_roi = wide_img(roi_wide);
		if (wide_img.empty() || narrow_img.empty())
			break;
		mkpts = lightglue.getMatchPoints(wide_img_roi, narrow_img, roi_wide, roi_narrow);
		cv::Mat M;
		cv::Mat resultImg;

		if (mkpts[0].size() > 10)
		{
			lightglue.setThreshold(0.00, 0.5);


			cv::Mat Mask;
			double ransac_threshold = 25;
			M = cv::findHomography(mkpts[1], mkpts[0], cv::RHO, ransac_threshold, Mask);
			int inliner_count = cv::countNonZero(Mask);

			std::vector<cv::Point2f> dst_points = { cv::Point2f(0, 0), cv::Point2f(narrow_img.cols, 0), cv::Point2f(narrow_img.cols, narrow_img.rows), cv::Point2f(0, narrow_img.rows) };
			cv::perspectiveTransform(dst_points, dst_points, M);
			std::string output_dir = remove_extension_c(wide_video_flie.c_str());
			std::cout << M << std::endl;
			std::string output_json_name = "output\\"+ output_dir +"\\homographies\\"+ output_dir + "_" + std::to_string(output_count) + ".json";
			std::string output_wide_name = "output\\" + output_dir + "\\images_wide\\" + output_dir + "_wide_" + std::to_string(output_count) + ".jpg";
			std::string output_narrow_name = "output\\" + output_dir + "\\images_narrow\\" + output_dir +"_narrow_" + std::to_string(output_count) + ".jpg";
			//output the json file 
			int res_out = WriteJsonFile(output_json_name.c_str(), M);

			cv::imwrite(output_wide_name, wide_img);
			cv::imwrite(output_narrow_name, narrow_img);


			edge_points.Push_Back(dst_points);

			roi_wide = cv::boundingRect(dst_points);
			std::cout << roi_wide << std::endl;
			roi_wide = roi_wide & roi_narrow;
			cv::warpPerspective(narrow_img, resultImg, M, cv::Size(wide_img.cols, wide_img.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
			warped_images.Push_Back(resultImg.clone());
			cv::Mat resultImg_mask = resultImg.clone();
			cv::Mat mask;
			if (resultImg_mask.channels() == 1) {
				mask = (resultImg_mask == 0);
			}
			else {
					
				std::vector<cv::Mat> channels;
				cv::split(resultImg_mask, channels);
				mask = (channels[0] == 0);
				for (size_t i = 1; i < channels.size(); ++i) {
					mask = mask & (channels[i] == 0);
				}
			}
			mask.convertTo(mask, CV_8UC1, 255); 
			wide_img.copyTo(resultImg_mask, mask);

			resultImg = resultImg.zeros(cv::Size(resultImg.cols, resultImg.rows), CV_8UC3);
			for (size_t i = 0; i < warped_images.Size(); i++)
			{
				cv::addWeighted(resultImg, 1, warped_images.queue[i], 0.5 / warped_images.Size(), 1, resultImg);

			}
			cv::addWeighted(wide_img, 0.5, resultImg, 1, 1, resultImg);
			inliner_sum += inliner_count;
			outliner_sum += mkpts[0].size() - inliner_count;
			for (int i = 0; i < edge_points.queue.size(); i++)
			{
				int color_int = 255 * i / edge_points.queue.size();
				cv::Scalar color = cv::Scalar(color_int, 255 - color_int, 255 - color_int);
				for (int j = 0; j < edge_points.queue[i].size(); j++)
					cv::line(resultImg, edge_points.queue[i][j], edge_points.queue[i][(j + 1) % 4], color, 1); 
			}
			/*
			* Calculate the changes in edge points and start calculating the average value after 20 frames
			*/
			if (last_edge_points.size() > 0 && count > 40)
			{
				double max_distance = 0;
				for (size_t i = 0; i < last_edge_points.size(); i++)
				{

					cv::Point2f p1 = last_edge_points[i];
					cv::Point2f p2 = dst_points[i];
					double distance = cv::norm(p1 - p2);
					max_distance = std::max(max_distance, distance);
				}
				distance_sum += max_distance;
				avg_edge_point_distance = distance_sum / (count - 40);
			}
			last_edge_points = dst_points;
			cv::putText(resultImg, "inliner_count: " + std::to_string(inliner_count), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
			cv::putText(resultImg, "outliner_count: " + std::to_string(mkpts[0].size() - inliner_count), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			cv::putText(resultImg, "frame_id: " + std::to_string(count), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
			cv::putText(resultImg, "RHO:" + std::to_string(ransac_threshold), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
			cv::putText(resultImg, "mean_inliner: " + std::to_string(inliner_sum / count), cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
			cv::putText(resultImg, "mean_outliner: " + std::to_string(outliner_sum / count), cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			cv::putText(resultImg, "mean_edge_point_distance: " + std::to_string(avg_edge_point_distance), cv::Point(10, 210), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

			cv::imwrite("output\\"+ output_dir + "\\res_match\\res_match" + std::to_string(count) + ".jpg", resultImg);
			cv::waitKey(1);
			++count;

			/*
			* Calculate the variation of M
			*/
			if (!last_M.empty())
			{
				static double norm_sum = 0;

				cv::Mat diff;
				cv::absdiff(last_M, M, diff);
				double norm = cv::norm(diff, cv::NORM_L2);
				norm_sum += norm;
				if (count % 50 == 0)
				{
					std::cout << "mean_diff:  " << norm_sum / 50 << std::endl;
					norm_sum = 0;
				}
			}
			last_M = M;

		}
		else
		{
			lightglue.setThreshold(0.00, 0.3);
			roi_wide = cv::Rect2f(0, 0, wide_img.cols, wide_img.rows);
			std::cout << "Number of feature points:  " << mkpts[0].size() << std::endl;
		}
		output_count++;
	}
}