#pragma once
/*
*Copyright (c) 2020-,  Ningbo Institute of Materials Technology and Engineering, Chinese Academy of Sciences
*@file LightGLue.h
*@brief Using LightGlue for feature point matching
*@author WuJiaZong
*@date 2024/06/05
* @version 1.1
*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <future>

#include "FixedSizeQueue.h"

class LightGlue
{
public:
	/*
	* @name LightGlue
	* @brief LightGlue **Constructor, initialize inference environment
	* @param extractor_threshold **Confidence threshold for feature extractor (superPoint)
	* @param lightglue_threshold **Confidence threshold for feature matching (lightGlue)
	* @param model_path_superpoint **SuperPoint's model path
	* @param model_path_lightglue **lightGlue's model path
	*/
	LightGlue(float extractor_threshold, float lightglue_threshold, std::string model_path_superpoint, std::string model_path_lightglue);
	/*
	* @name getMatchPoints
	* @brief **Obtain matching points from two images
	* @param img0 **The first picture
	* @param img1 **The second picture
	* @param roi0 **The region of interest in the first image
	* @param roi_img0_screen **The position of the first image on the screen
	* @return std::vector<std::vector<cv::Point2f> > **Matching point container for two images
	*/
	std::vector<std::vector<cv::Point2f> > getMatchPoints(const cv::Mat& img0, const cv::Mat& img1, const cv::Rect2f& roi0, cv::Rect2f& roi_img0_screen);

	/*
	* @name setThreshold
	* @brief **Set the confidence threshold for the feature extractor (superPoint) and the confidence threshold for the feature matching (lightGlue)
	* @param extractor_threshold **Confidence threshold for feature extractor (superPoint)
	* @param lightglue_threshold **Confidence threshold for feature matching (lightGlue)
	*/
	void setThreshold(float extractor_threshold, float lightglue_threshold);

	/*
	* @name setDefaultThreshold
	* @brief **The confidence threshold for initializing the feature extractor (superPoint) and the confidence threshold for feature matching (lightGlue)
	*/
	void setDefaultThreshold();



private:
	int inpWidth;
	int inpHeight;
	float m_extractor_threshold;
	float m_lightglue_threshold;

	float default_extractor_threshold;
	float default_lightglue_threshold;

	std::vector<float> input_image_0;
	std::vector<float> input_image_1;
	std::vector<float> image0_scalar;
	std::vector<float> image1_scalar;

	std::vector<float> img_preProcess(cv::Mat& img); //Image preprocessing, converting the image into the input format of superPoint and returning the change ratio of the image
	std::vector<float> normalize_keypoints(std::vector<cv::Point2f>& kpts, float w, float h);

	void superPoint_postProcess(std::vector<cv::Point2f>& kpts, std::vector<float>& scores, std::vector<std::vector<float> >& desc, float extractor_threshold);
	void lightglue_postProcess(std::vector<cv::Point2f>& mkpts0, std::vector<cv::Point2f>& mkpts1, std::vector<float> scalar0, std::vector<float> scalar1);//LightGlue post-processing maps the points output by onnx back to the original image size

	Ort::Env env_superpoint0 = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "SUPERPOINT"); 
	Ort::Env env_superpoint1 = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "SUPERPOINT");
	Ort::Env env_lightglue = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "LIGHTGLUE");

	std::shared_ptr<Ort::Session> superpoint_ort_session0; //The first image feature point extraction inference context
	std::shared_ptr<Ort::Session> superpoint_ort_session1; //The second image feature point extraction inference context
	std::shared_ptr<Ort::Session> lightglue_ort_session; //Match feature points in two images to infer context

	std::vector<float> normalize(const cv::Mat& img); 

	Ort::SessionOptions sessionOptions_superpoint0 = Ort::SessionOptions();
	Ort::SessionOptions sessionOptions_superpoint1 = Ort::SessionOptions();
	Ort::SessionOptions sessionOptions_lightglue = Ort::SessionOptions();

	std::vector<std::string> superpoint_input_names;
	std::vector<std::string> lightglue_input_names;
	std::vector<std::string> superpoint_output_names;
	std::vector<std::string> lightglue_output_names;
	std::vector<std::vector<int64_t>> superpoint_input_node_dims; // >=1 inputs
	std::vector<std::vector<int64_t>> lightglue_input_node_dims; // >=1 inputs
	std::vector<std::vector<int64_t>> superpoint_output_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> lightglue_output_node_dims; // >=1 outputs
	FixedSizeQueue< std::vector<cv::Point2f> > match_points_queue; //Match point queue, used to cache match points
	FixedSizeQueue<std::vector<std::vector <float> > >match_desc_queue; //Match descriptor queue, used to cache match descriptors
	FixedSizeQueue<std::vector<float> > match_score_queue; //Match score queue, used to cache match scores
};

