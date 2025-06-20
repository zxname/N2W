#include "LightGlue.h"
#include <opencv2//highgui.hpp>
LightGlue::LightGlue(float extractor_threshold, float lightglue_threshold, std::string model_path_superpoint, std::string model_path_lightglue)
{
	this->m_extractor_threshold = extractor_threshold;
	this->m_lightglue_threshold = lightglue_threshold;
	this->default_extractor_threshold = extractor_threshold;
	this->default_lightglue_threshold = lightglue_threshold;
	match_points_queue.Set_Max_Size(40);
	match_desc_queue.Set_Max_Size(40);
	match_score_queue.Set_Max_Size(40);

	auto available_str = Ort::GetAvailableProviders();
	for (size_t i = 0; i < available_str.size(); i++)
	{
		std::cout << "Available providers: " << available_str[i] << std::endl;

	}
	std::wstring widestr_model_path_superpoint = std::wstring(model_path_superpoint.begin(), model_path_superpoint.end());
	std::wstring widestr_model_path_lightglue = std::wstring(model_path_lightglue.begin(), model_path_lightglue.end());
	OrtStatus* status_superPoint0 = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_superpoint0, 0);
	OrtStatus* status_superPoint1 = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_superpoint1, 0);
	OrtStatus* status_lightglue = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_lightglue, 0);

	sessionOptions_superpoint0.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	sessionOptions_superpoint1.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	sessionOptions_lightglue.SetGraphOptimizationLevel(ORT_ENABLE_ALL);


	this->superpoint_ort_session0 = std::make_shared<Ort::Session>(this->env_superpoint0, widestr_model_path_superpoint.c_str(), sessionOptions_superpoint0);
	this->superpoint_ort_session1 = std::make_shared<Ort::Session>(this->env_superpoint1, widestr_model_path_superpoint.c_str(), sessionOptions_superpoint1);
	this->lightglue_ort_session = std::make_shared<Ort::Session>(this->env_lightglue, widestr_model_path_lightglue.c_str(), sessionOptions_lightglue);

	size_t numInputNodes_superPoint = superpoint_ort_session0->GetInputCount();
	size_t numOutputNodes_superPoint = superpoint_ort_session0->GetOutputCount();
	size_t numInputNodes_lightGlue = lightglue_ort_session->GetInputCount();
	size_t numOutputNodes_lightGlue = lightglue_ort_session->GetOutputCount();

	Ort::AllocatorWithDefaultOptions allocator;

	/******************************SuperPoint initialization***********************/

	for (size_t i = 0; i < numInputNodes_superPoint; i++)
	{
		auto input_name = superpoint_ort_session0->GetInputNameAllocated(i, allocator);

		std::string input_name_string;
		input_name_string.assign(input_name.get());
		this->superpoint_input_names.push_back(input_name_string);

		Ort::TypeInfo input_type_info = superpoint_ort_session0->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		this->superpoint_input_node_dims.emplace_back(input_dims);
	}

	for (size_t i = 0; i < numOutputNodes_superPoint; i++)
	{
		auto output_name = superpoint_ort_session0->GetOutputNameAllocated(i, allocator);
		std::string output_name_string;
		output_name_string.assign(output_name.get());
		this->superpoint_output_names.push_back(output_name_string);

		Ort::TypeInfo output_type_info = superpoint_ort_session0->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		superpoint_output_node_dims.push_back(output_dims);
	}
	this->inpHeight = superpoint_input_node_dims[0][2];
	this->inpWidth = superpoint_input_node_dims[0][3];


	/**********************lightGlue initialization************************/

	for (size_t i = 0; i < numInputNodes_lightGlue; i++)
	{
		auto input_name = lightglue_ort_session->GetInputNameAllocated(i, allocator);

		std::string input_name_string;
		input_name_string.assign(input_name.get());
		this->lightglue_input_names.push_back(input_name_string);

		Ort::TypeInfo input_type_info = lightglue_ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		this->lightglue_input_node_dims.emplace_back(input_dims);
	}

	for (size_t i = 0; i < numOutputNodes_lightGlue; i++)
	{
		auto output_name = lightglue_ort_session->GetOutputNameAllocated(i, allocator);
		std::string output_name_string;
		output_name_string.assign(output_name.get());
		this->lightglue_output_names.push_back(output_name_string);

		Ort::TypeInfo output_type_info = lightglue_ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		lightglue_output_node_dims.push_back(output_dims);
	}



}

std::vector<std::vector<cv::Point2f> > LightGlue::getMatchPoints(const cv::Mat& img0, const cv::Mat& img1, const cv::Rect2f& roi0, cv::Rect2f& roi_img0_screen)
{
	std::vector<std::vector<cv::Point2f> > matchedPoints;
	cv::Mat img0_temp_show = img0.clone();
	cv::Mat img1_temp_show = img1.clone();
	cv::Mat img0_temp_show_out = img0.clone();
	cv::Mat img1_temp_show_out = img1.clone();

	cv::Mat img0_temp = img0.clone();
	cv::Mat img1_temp = img1.clone();
	/////////////////////////////////////////////Image preprocessing
	this->image0_scalar = img_preProcess(img0_temp);
	this->image1_scalar = img_preProcess(img1_temp);
	cv::resize(img0_temp_show, img0_temp_show, cv::Size(this->inpWidth, this->inpHeight), 0, 0, cv::INTER_AREA);
	cv::resize(img1_temp_show, img1_temp_show, cv::Size(this->inpWidth, this->inpHeight), 0, 0, cv::INTER_AREA);

	///////////////////////////////////////////
	std::array<std::int64_t, 4> input_shape_0;
	std::array<std::int64_t, 4> input_shape_1;
	if (this->inpHeight == -1 && this->inpWidth == -1)
	{
		input_shape_0 = { 1, img0_temp.channels(), img0.rows, img0.cols };
		input_shape_1 = { 1, img1_temp.channels(), img1.rows, img1.cols };
	}
	else
	{
		input_shape_0 = { 1, img0_temp.channels(), this->inpHeight, this->inpWidth };
		input_shape_1 = { 1, img1_temp.channels(), this->inpHeight, this->inpWidth };
	}

	auto allocator_info_superPoint0 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto allocator_info_superPoint1 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	size_t size_0 = img0_temp.rows * img0_temp.cols * img0_temp.channels();
	size_t size_1 = img1_temp.rows * img1_temp.cols * img1_temp.channels();

	Ort::Value input_tensor_0 = Ort::Value::CreateTensor<float>(allocator_info_superPoint0, (float*)img0_temp.data, size_0, input_shape_0.data(), input_shape_0.size());
	Ort::Value input_tensor_1 = Ort::Value::CreateTensor<float>(allocator_info_superPoint1, (float*)img1_temp.data, size_1, input_shape_1.data(), input_shape_1.size());
	const auto option = Ort::RunOptions{ nullptr };

	std::vector<char*> superpoint_input_names_c;
	std::vector<char*> superpoint_output_names_c;
	for (size_t i = 0; i < superpoint_input_names.size(); i++)
	{
		superpoint_input_names_c.push_back((char*)superpoint_input_names[i].c_str());
	}

	for (size_t i = 0; i < superpoint_output_names.size(); i++)
	{
		superpoint_output_names_c.push_back((char*)superpoint_output_names[i].c_str());
	}

	/*********************************Using asynchronous reasoning methods********************************************/
	std::future<std::vector<Ort::Value> > future_superPoint_0 = std::async(std::launch::async, [&]() {
		std::vector<Ort::Value> ort_outputs_0 = superpoint_ort_session0->Run(Ort::RunOptions{ nullptr }, (char**)superpoint_input_names_c.data(), & input_tensor_0, 1, (char**)superpoint_output_names_c.data(), superpoint_output_names.size());   
		return ort_outputs_0;
		});
	std::future<std::vector<Ort::Value> > future_superPoint_1 = std::async(std::launch::async, [&]() {
		std::vector<Ort::Value> ort_outputs_0 = superpoint_ort_session0->Run(Ort::RunOptions{ nullptr }, (char**)superpoint_input_names_c.data(), & input_tensor_1, 1, (char**)superpoint_output_names_c.data(), superpoint_output_names.size());   
		return ort_outputs_0;
		});
	std::vector<Ort::Value> ort_outputs_0 = future_superPoint_0.get();//Obtain reasoning results
	std::vector<Ort::Value> ort_outputs_1 = future_superPoint_1.get();//Obtain reasoning results

	
	/*********************************Obtain two key points for the images********************************************/
	auto kpts0_tensor_info_output = ort_outputs_0[0].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> kpts0_tensorShape = kpts0_tensor_info_output.GetShape();

	auto kpts1_tensor_info_output = ort_outputs_1[0].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> kpts1_tensorShape = kpts1_tensor_info_output.GetShape();

	int64_t* kpts0_int64_p = ort_outputs_0[0].GetTensorMutableData<int64_t>();
	int64_t* kpts1_int64_p = ort_outputs_1[0].GetTensorMutableData<int64_t>();

	size_t kpts0_size = 1;
	for (auto shape : kpts0_tensorShape)
	{
		kpts0_size *= shape;
	}

	size_t kpts1_size = 1;
	for (auto shape : kpts1_tensorShape)
	{
		kpts1_size *= shape;
	}
	kpts0_size /= 2;
	kpts1_size /= 2;
	std::vector<cv::Point2f> kpts0(kpts0_size);
	std::vector<cv::Point2f> kpts1(kpts1_size);
	for (size_t i = 0; i < kpts0_size; i++)//Save feature points of image0
	{
		kpts0[i] = cv::Point2f(*(kpts0_int64_p + i * 2), *(kpts0_int64_p + i * 2 + 1));
	}
	for (size_t i = 0; i < kpts1_size; i++)//Save feature points of image1
	{
		kpts1[i] = cv::Point2f(*(kpts1_int64_p + i * 2), *(kpts1_int64_p + i * 2 + 1));
	}

	/******************************************************************************************************/
	/*****************************Obtain two image scores*****************************************************/
	auto scores0_tensor_info_output = ort_outputs_0[1].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> scores0_tensorShape = scores0_tensor_info_output.GetShape();

	auto scores1_tensor_info_output = ort_outputs_1[1].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> scores1_tensorShape = scores1_tensor_info_output.GetShape();

	float* scores0_float_p = ort_outputs_0[1].GetTensorMutableData<float>();
	float* scores1_float_p = ort_outputs_0[1].GetTensorMutableData<float>();

	size_t scores0_size = 1;
	for (auto shape : scores0_tensorShape)
	{
		scores0_size *= shape;
	}

	size_t scores1_size = 1;
	for (auto shape : scores1_tensorShape)
	{
		scores1_size *= shape;
	}
	std::vector<float> scores0;
	std::vector<float> scores1;

	for (size_t i = 0; i < scores0_size; i++)//Save the score of image0
	{
		scores0.emplace_back(*(scores0_float_p + i));
	}
	for (size_t i = 0; i < scores1_size; i++)//Save the score of image1
	{
		scores1.emplace_back(*(scores1_float_p + i));
	}
	/********************************************Obtain descriptors of feature points in two images**********************************************************/
	auto desc0_tensor_info_output = ort_outputs_0[2].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> desc0_tensorShape = desc0_tensor_info_output.GetShape();

	auto desc1_tensor_info_output = ort_outputs_1[2].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> desc1_tensorShape = desc1_tensor_info_output.GetShape();

	float* desc0_float_p = ort_outputs_0[2].GetTensorMutableData<float>();
	float* desc1_float_p = ort_outputs_1[2].GetTensorMutableData<float>();

	size_t desc0_size = 1;
	size_t desc0_dim = 1;
	for (auto shape : desc0_tensorShape)
	{
		desc0_size *= shape;
	}
	desc0_dim = desc0_tensorShape[2];
	size_t desc1_size = 1;
	size_t desc1_dim = 1;

	for (auto shape : desc1_tensorShape)
	{
		desc1_size *= shape;
	}
	desc1_dim = desc1_tensorShape[2];
	std::vector<std::vector<float> > desc0;
	std::vector<std::vector<float> > desc1;

	for (size_t i = 0; i < kpts0_size; i++)//Save the feature descriptor of image0
	{
		std::vector<float> desc_pt;
		for (size_t j = 0; j < desc0_dim; j++)
			desc_pt.emplace_back(*(desc0_float_p + i * desc0_dim + j));
		desc0.emplace_back(desc_pt);
	}
	for (size_t i = 0; i < kpts1_size; i++)//Save the feature descriptor of image1
	{
		std::vector<float> desc_pt;
		for (size_t j = 0; j < desc1_dim; j++)
			desc_pt.emplace_back(*(desc1_float_p + i * desc1_dim + j));
		desc1.emplace_back(desc_pt);

	}
	for (const auto& kpt0 : kpts0)
	{
		cv::circle(img0_temp_show, cv::Point2f(kpt0.x, kpt0.y), 2, cv::Scalar(0, 255, 0), -1);
	}
	
	desc0_size = desc0_dim * desc0.size();
	desc1_size = desc1_dim * desc1.size();
	/**************************************Post processing of feature points**********************************************************/
	superPoint_postProcess(kpts0, scores0, desc0, this->m_extractor_threshold);
	superPoint_postProcess(kpts1, scores1, desc1, 0);
	std::cout << "kpts0 size: " << kpts0.size() << std::endl;
	std::cout << "kpts1 size: " << kpts1.size() << std::endl;
	for (const auto& kpt1 : kpts1)
	{
		cv::circle(img1_temp_show, cv::Point2f(kpt1.x, kpt1.y), 2, cv::Scalar(0, 0, 255), -1);
	}

	/**************************************Feature point matching**********/
	/**************************************Lightglue pre-processing************************************/
	//Normalization of feature points
	auto kpts0_norm = normalize_keypoints(kpts0, img0.cols, img0.rows);
	auto kpts1_norm = normalize_keypoints(kpts1, img1.cols, img1.rows);
	std::vector<float> desc0_flatten(desc0_size);
	std::vector<float> desc1_flatten(desc1_size);
	for (size_t i = 0; i < desc0.size(); i++)
	{
		for (size_t j = 0; j < desc0[i].size(); j++)
			desc0_flatten[i * desc0[i].size() + j] = desc0[i][j];
	}
	for (size_t i = 0; i < desc1.size(); i++)
	{
		for (size_t j = 0; j < desc1[i].size(); j++)
			desc1_flatten[i * desc1[i].size() + j] = desc1[i][j];
	}

	/***********************Create Tensor for Reasoning**********************/
	std::array<int64_t, 3> kpts0_shape{ this->lightglue_input_node_dims[0][0], kpts0.size(), this->lightglue_input_node_dims[0][2]};
	std::array<int64_t, 3> kpts1_shape{ this->lightglue_input_node_dims[1][0], kpts1.size(), this->lightglue_input_node_dims[1][2]};
	std::array<int64_t, 3> desc0_shape{ this->lightglue_input_node_dims[2][0], kpts0.size(), this->lightglue_input_node_dims[2][2]};
	std::array<int64_t, 3> desc1_shape{ this->lightglue_input_node_dims[3][0], kpts1.size(), this->lightglue_input_node_dims[3][2]};

	auto allocator_info_kpts0 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto allocator_info_kpts1 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto allocator_info_desc0 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto allocator_info_desc1 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Ort::Value input_tensor_kpts0 = Ort::Value::CreateTensor<float>(allocator_info_kpts0, kpts0_norm.data(), kpts0_norm.size(), kpts0_shape.data(), kpts0_shape.size());
	Ort::Value input_tensor_kpts1 = Ort::Value::CreateTensor<float>(allocator_info_kpts1, kpts1_norm.data(), kpts1_norm.size(), kpts1_shape.data(), kpts1_shape.size());
	Ort::Value input_tensor_desc0 = Ort::Value::CreateTensor<float>(allocator_info_desc0, desc0_flatten.data(), desc0_flatten.size(), desc0_shape.data(), desc0_shape.size());
	Ort::Value input_tensor_desc1 = Ort::Value::CreateTensor<float>(allocator_info_desc1, desc1_flatten.data(), desc1_flatten.size(), desc1_shape.data(), desc1_shape.size());

	std::vector<char*> lightglue_input_names_c;
	std::vector<char*> lightglue_output_names_c;
	for (size_t i = 0; i < lightglue_input_names.size(); i++)
	{
		lightglue_input_names_c.push_back((char*)lightglue_input_names[i].c_str());
	}

	for (size_t i = 0; i < lightglue_output_names.size(); i++)
	{
		lightglue_output_names_c.push_back((char*)lightglue_output_names[i].c_str());
	}
	Ort::Value lightglue_input[] = { std::move(input_tensor_kpts0), std::move(input_tensor_kpts1), std::move(input_tensor_desc0), std::move(input_tensor_desc1) };
	/*****************************************infer********************************/
	std::vector<Ort::Value> ort_outputs_lightglue = lightglue_ort_session->Run(Ort::RunOptions{ nullptr }, lightglue_input_names_c.data(), lightglue_input, lightglue_input_names_c.size(), (char**)lightglue_output_names_c.data(), lightglue_output_names_c.size());   // 开始推理

	auto matches_tensor_info_output = ort_outputs_lightglue[0].GetTensorTypeAndShapeInfo();
	auto mscores_tensor_info_output = ort_outputs_lightglue[1].GetTensorTypeAndShapeInfo();

	std::vector<int64_t> matches_tensorShape = matches_tensor_info_output.GetShape();
	std::vector<int64_t> mscores_tensorShape = mscores_tensor_info_output.GetShape();
	std::vector<std::vector<int64_t> > matches;
	std::vector<float> mscores;
	int64_t* matches_int64_p = ort_outputs_lightglue[0].GetTensorMutableData<int64_t>();
	float* matches_float_p = ort_outputs_lightglue[1].GetTensorMutableData<float>();

	for (size_t i = 0; i < matches_tensorShape[0]; i++)
	{
		if (matches_float_p[i] > this->m_lightglue_threshold)
		{
			std::vector<int64_t> matches_pair(matches_tensorShape[1]);
			matches_pair[0] = matches_int64_p[i * 2];
			matches_pair[1] = matches_int64_p[i * 2 + 1];
			matches.emplace_back(matches_pair);
			mscores.emplace_back(matches_float_p[i]);
		}
	}
	std::vector <cv::Point2f> kpts0_matches(matches.size());
	std::vector <cv::Point2f> kpts1_matches(matches.size());
	std::vector <std::vector<float> > desc0_matches(matches.size());
	std::vector <std::vector<float> > desc1_matches(matches.size());
	std::vector <float> score0_matches(matches.size());
	std::vector <float> score1_matches(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		kpts0_matches[i] = kpts0[matches[i][0]];
		kpts1_matches[i] = kpts1[matches[i][1]];
		desc0_matches[i] = desc0[matches[i][0]];
		desc1_matches[i] = desc1[matches[i][1]];
		score0_matches[i] = scores0[matches[i][0]];
		score1_matches[i] = scores1[matches[i][1]];
	}
	/*****************************************Post processing********************************/
	std::vector<cv::Point2f> kpts0_good;
	std::vector<std::vector<float> > desc0_good;
	std::vector<float> score0_good;
	lightglue_postProcess(kpts0_matches, kpts1_matches, this->image0_scalar, this->image1_scalar);
	for (auto& kpt0 : kpts0_matches)
	{
		kpt0.x = kpt0.x + roi0.x;
		kpt0.y = kpt0.y + roi0.y;
	}
	matchedPoints.emplace_back(kpts0_matches);
	matchedPoints.emplace_back(kpts1_matches);


	for (size_t i = 0; i < kpts0_matches.size(); i++)
	{
		if (mscores[i] > 0.7 && score1_matches[i] > 0.0)
		{
			kpts0_good.emplace_back(kpts0_matches[i]);
			desc0_good.emplace_back(desc1_matches[i]);
			score0_good.emplace_back(mscores[i]);
		}
	}

	match_points_queue.Push_Back(kpts0_good);
	match_desc_queue.Push_Back(desc0_good);
	match_score_queue.Push_Back(score0_good);
	cv::Mat background(cv::Size(1536, 2048), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::rectangle(background, cv::Rect(roi0.x, roi0.y, roi0.width, roi0.height), cv::Scalar(255, 0, 0), 2);
	
	for (int keypoint_index = 0; keypoint_index < matchedPoints[0].size(); ++keypoint_index)
	{
		cv::Scalar circle_color;


		if (mscores[keypoint_index] > 0.7)
		{
			circle_color = cv::Scalar(0, 255, 255);
			cv::circle(background, cv::Point2f(matchedPoints[0][keypoint_index].x, matchedPoints[0][keypoint_index].y), 2, circle_color, -1);
		cv:circle(img0_temp_show_out, cv::Point2f(matchedPoints[0][keypoint_index].x - roi0.x, matchedPoints[0][keypoint_index].y - roi0.y), 1, circle_color, -1);
			cv::circle(img1_temp_show_out, cv::Point2f(matchedPoints[1][keypoint_index].x, matchedPoints[1][keypoint_index].y), 5, circle_color, -1);
		}
	}
	//cv::namedWindow("background", cv::WINDOW_NORMAL);
	//cv::imshow("background", background);
	//cv::namedWindow("img1_temp_show_out", cv::WINDOW_NORMAL);
	//cv::imshow("img1_temp_show_out", img1_temp_show_out);
	//cv::namedWindow("img0_temp_show_out", cv::WINDOW_NORMAL);
	//cv::imshow("img0_temp_show_out", img0_temp_show_out);
	//cv::waitKey(0);
	//std::cout << "success" << std::endl;
	return matchedPoints;

}

void LightGlue::setThreshold(float extractor_threshold, float lightglue_threshold)
{
	this->m_extractor_threshold = extractor_threshold;
	this->m_lightglue_threshold = lightglue_threshold;
}

void LightGlue::setDefaultThreshold()
{
	this->m_extractor_threshold = this->default_extractor_threshold;
	this->m_lightglue_threshold = this->default_lightglue_threshold;
}

std::vector<float> LightGlue::img_preProcess(cv::Mat& img)
{
	std::vector<float> scales(2);
	img.convertTo(img, CV_32FC3);

	if (inpWidth == -1 && inpHeight == -1)
	{
		scales[0] = 1;
		scales[1] = 1;
	}
	else
	{
		scales[0] = (float)inpWidth / (float)img.cols;
		scales[1] = (float)inpHeight / (float)img.rows;
		cv::resize(img, img, cv::Size(this->inpWidth, this->inpHeight), 0, 0, cv::INTER_AREA);
	}
	img = img / 255.0;
	switch (img.channels())
	{
	case 1:
		break;
	case 3:
		cvtColor(img, img, cv::COLOR_BGR2GRAY);
		break;
	case 4:
		cvtColor(img, img, cv::COLOR_BGRA2GRAY);
		break;

	default:
		break;
	}
	return scales;
}

std::vector<float> LightGlue::normalize_keypoints(std::vector<cv::Point2f>& kpts, float w, float h)
{
	std::vector<float> kpts_normalized(kpts.size() * 2);
	if (inpWidth != -1 && inpHeight != -1)
	{
		w = this->inpWidth, h = this->inpHeight;
	}
	cv::Point2f shift(w / 2, h / 2);
	float scalar = std::max(w, h) / 2;
	size_t kpts_iter = 0;
	for (auto point : kpts)
	{
		point = (point - shift) / scalar;
		kpts_normalized[kpts_iter] = point.x;
		kpts_iter++;
		kpts_normalized[kpts_iter] = point.y;
		kpts_iter++;
	}
	return kpts_normalized;
}

void LightGlue::superPoint_postProcess(std::vector<cv::Point2f>& kpts, std::vector<float>& scores, std::vector<std::vector<float> >& desc, float extractor_threshold)
{
	auto it_kpts = kpts.begin();
	auto it_scores = scores.begin();
	auto it_desc = desc.begin();

	while (it_scores != scores.end())
	{
		if (*it_scores < extractor_threshold) {
			it_kpts = kpts.erase(it_kpts);
			it_scores = scores.erase(it_scores);
			it_desc = desc.erase(it_desc);
		}
		else {
			it_kpts++;
			it_scores++;
			it_desc++;
		}
	}
	kpts.shrink_to_fit();
	scores.shrink_to_fit();
	desc.shrink_to_fit();

}

void LightGlue::lightglue_postProcess(std::vector<cv::Point2f>& mkpts0, std::vector<cv::Point2f>& mkpts1, std::vector<float> scalar0, std::vector<float> scalar1)
{
	for (size_t i = 0; i < mkpts0.size(); i++)
	{
		mkpts0[i].x = (mkpts0[i].x + 0.5) / scalar0[0] - 0.5;
		mkpts0[i].y = (mkpts0[i].y + 0.5) / scalar0[1] - 0.5;
	}
	for (size_t i = 0; i < mkpts1.size(); i++)
	{
		mkpts1[i].x = (mkpts1[i].x + 0.5) / scalar1[0] - 0.5;
		mkpts1[i].y = (mkpts1[i].y + 0.5) / scalar1[1] - 0.5;
	}
}

std::vector<float> LightGlue::normalize(const cv::Mat& img)
{
	int row = img.rows;
	int col = img.cols;
	int channels = img.channels();
	std::vector<float> input_image(row * col * channels);
	for (int c = 0; c < channels; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<float>(i)[j];
				input_image[c * row * col + i * col + j] = pix;
			}
		}
	}
	return input_image;
}