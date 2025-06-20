# Narrow to Wide Annotation

Our N2W provides C++ sample code for multi-frame matching, demonstrating the generation of both wide-FOV and narrow-FOV images from individual frames, along with JSON file output of homography matrices. Additionally, we provide Python implementation for annotation box mapping, which utilizes the generated homography JSON files and VOC-format detection files (from object detection methods on narrow-FOV images) to produce wide-FOV annotation files.



## File Tree:

***pythonForAnnotation***

└─dataset_root
    ├─1
    │  ├─homographies
    │  ├─images_narrow
    │  ├─images_wide
    │  └─labels_narrow
    ├─10
    │  ├─homographies
    │  ├─images_narrow
    │  ├─images_wide
    │  └─labels_narrow
    ├─11
    │  ├─homographies
    │  ├─images_narrow
    │  ├─images_wide
    │  └─labels_narrow
    ......
    
 ***C++ multi-frame match***
 └─output_root
    ├─wide_demo1
    │  ├─homographies
    │  ├─images_narrow
    │  ├─images_wide
    │  └─res_match
    ├─wide_demo2
    │  ├─homographies
    │  ├─images_narrow
    │  ├─images_wide
    │  └─res_match
    ├─wide_demo2
    │  ├─homographies
    │  ├─images_narrow
    │  ├─images_wide
    │  └─res_match
    ......
## Requirements
***N2W Annotation (python)***

   Python >= 3.9

**Dependencies:** 

    -- numpy==1.24.3
    -- opencv-contrib-python==4.10.0.84
    -- opencv-python==4.9.0.80
    -- pandas==2.2.1
    -- pillow==10.3.0
    -- tqdm==4.66.4
    -- matplotlib==3.8.4
  
 ***N2W multiFrame Register (C++)[code](https://github.com/zxname/N2W/tree/multiFrameRegister)***
 

 
 MSVC==v143
 
 cuda==11.8
 
 cudnn==8.5.0
 
 **Dependencies:**
 
    -- opencv==3.4.12
    -- onnxruntime_gpu==1.16.3
    -- json-nlohmann

## Installation

​	1.Clone the repository

​	2.Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start
1. Utilize the C++ multi-frame registration program to generate wide-FOV frames, narrow-FOV frames, and corresponding homography matrices. Then apply object detection algorithms to generate annotation files, forming the dataset list for pythonForAnnotation.
2. We also peovide some sample data in the '**asset**' folder

```bash
python annotate.py --N2W_dataset_root your_dataset_root_dir --dataset_output dataset_output_dir --outputName prefix_ --isSaveData True --viewResult True
```

