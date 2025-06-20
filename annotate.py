#encoding:utf-8
import random
import time
import os
import glob
import cv2
import numpy as np
import argparse
import json
import tqdm
from convex_polygon_intersection import intersect, _sort_vertices_anti_clockwise_and_remove_duplicates

from PIL import Image
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path

params = cv2.TrackerKCF.Params()
params.detect_thresh = 0.05

#
# function: Print matching points
#
def printMatchesPoints(image0, image1, points0, points1):
    image0_copy = image0.copy()
    image1_copy = image1.copy()
    if (image0.shape[0] > image1.shape[0]):
        image1_copy = cv2.copyMakeBorder(image1, 0, image0.shape[0] - image1.shape[0], 0, 0,
                                         cv2.BORDER_CONSTANT)
    else:
        image0_copy = cv2.copyMakeBorder(image0, 0, image1.shape[0] - image0.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT)
    res = cv2.hconcat([image0_copy, image1_copy])

    for index in range(0,len(points0)):
        color = [int(255*random.random()), int(255*random.random()),int(255*random.random())]
        point0 = [int(points0[index][0]), int(points0[index][1])]
        point1 = [int(points1[index][0]), int(points1[index][1])]

        point1_res = [point1[0]+wideFovImage.shape[1], point1[1]]

        cv2.circle(res, point0, 3, color, 3)
        cv2.circle(res, point1_res, 3, color, 3)
        cv2.line(res,point0, point1_res, color, 2)
    return res

# Get object names and bounding boxes from a single XML path
def getObjectsFromXml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = []
    xyxys = []
    for object in root.findall("object"):
        names.append(object.find("name").text)
        _ = object.keys()
        bndbox = object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        xyxys.append([xmin,ymin,xmax,ymax])
    return names, xyxys


## Check if xyxy is inside the polygon
def isXyxyInPolygon(contour, xyxy):
    point0 = (int(xyxy[0]), int(xyxy[1])) #tl
    point1 = (int(xyxy[2]), int(xyxy[1])) #tr
    point2 = (int(xyxy[2]), int(xyxy[3])) #br
    point3 = (int(xyxy[0]), int(xyxy[3])) #bt
    polygon1 = _sort_vertices_anti_clockwise_and_remove_duplicates([point0, point1, point2, point3])
    polygon2 = _sort_vertices_anti_clockwise_and_remove_duplicates(contour[0])
    polygon3 = intersect(polygon1, polygon2)#交集
    if len(polygon3):
        polygon3 = np.array(polygon3,dtype=np.float32)
        area_intersect = cv2.contourArea(polygon3)
        polygon1 = np.array(polygon1, dtype=np.float32)
        area_obj = cv2.contourArea(polygon1)
        if(area_intersect / area_obj) > 0.5:
            return True
    return False

# Remove rectangles inside the polygon
def getObjectsIsNotInPolyon(names, xyxys, contour):
    names_new = []
    xyxys_new = []
    for name, xyxy in zip(names, xyxys):
        if not isXyxyInPolygon(contour, xyxy):
            names_new.append(name)
            xyxys_new.append(xyxy)
    return names_new, xyxys_new


# Generate mask based on the coordinates of the bounding box
def xyxys_warp(M, xyxys):
    xyxys_new = []
    for xyxy in xyxys:
        points_np = np.array([[[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]],[xyxy[0], xyxy[3]]]])
        points_np = points_np.reshape(1, -1, 2).astype(np.float32)
        points_np_warped = cv2.perspectiveTransform(points_np, M)
        points_np_warped = points_np_warped.astype(np.int32)
        xmin = min(points_np_warped[0][0][0], points_np_warped[0][1][0], points_np_warped[0][2][0], points_np_warped[0][3][0])
        ymin = min(points_np_warped[0][0][1], points_np_warped[0][1][1], points_np_warped[0][2][1], points_np_warped[0][3][1])
        xmax = max(points_np_warped[0][0][0], points_np_warped[0][1][0], points_np_warped[0][2][0], points_np_warped[0][3][0])
        ymax = max(points_np_warped[0][0][1], points_np_warped[0][1][1], points_np_warped[0][2][1], points_np_warped[0][3][1])
        if (xmax - xmin) == 0 or (ymax - ymin) == 0:
            continue
        xyxys_new.append([xmin, ymin, xmax, ymax])
    return xyxys_new

#
# Beautify XML file
#
def prettyXml(element, indent, newline, level = 0): # element is the incoming Element class, indent for indentation, newline for line breaks
    if element:  # Check if element has child elements
        if element.text == None or element.text.isspace(): # If element's text has no content
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element) # Convert element to list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1): # If not the last element in the list, the next line is the start of the same level element, indentation should be consistent
            subelement.tail = newline + indent * (level + 1)
        else:  # If it is the last element in the list, the next line is the end of the parent element, indentation should be one less
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level = level + 1) # Recursive operation on child elements

# Add object to XML file
def add_object_xml(root, object_name, bbox):
    object_xml = ET.SubElement(root, "object")
    name = ET.SubElement(object_xml, "name")
    name.text = object_name
    pose = ET.SubElement(object_xml, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(object_xml, "truncated")
    truncated.text = "0"
    difficult = ET.SubElement(object_xml, "difficult")
    difficult.text = "0"
    bndbox = ET.SubElement(object_xml, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(bbox[0])
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(bbox[1])
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(bbox[2])
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(bbox[3])
    return root

# Generate VOC XML file

def create_voc_xml(filename, width, height, depth, object_name, bbox):
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = "images"

    filename_xml = ET.SubElement(root, "filename")
    filename_xml.text = filename

    path_xml = ET.SubElement(root, "path")
    path_xml.text = filename

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    width_xml = ET.SubElement(size, "width")
    width_xml.text = str(width)
    height_xml = ET.SubElement(size, "height")
    height_xml.text = str(height)
    depth_xml = ET.SubElement(size, "depth")
    depth_xml.text = str(depth)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    add_object_xml(root,object_name,bbox)
    return root


parser = argparse.ArgumentParser(description='imageMatchForDataGenerate')

parser.add_argument('--N2W_dataset_root', type=str, default=r"assets\example_data\construction_site")
parser.add_argument('--dataset_output', type=str, default=r"output_dataset_construction_site")

parser.add_argument('--RandomDataNum', type=int, default=10, help='The total number of randomly generated images')

parser.add_argument('--outputName', type=str, default=r"construction_site_", help="The prefix of the output file")
parser.add_argument('--isSaveData', default=False, help="Whether to save data")
parser.add_argument('--viewResult', default=False, help="Use imshow to view the running results in real time")
# parser.add_argument('--matchThreshold', type=float, default=0.5)
args = parser.parse_args()  


if __name__ == '__main__':
    viewResult = args.viewResult
    N2W_dataset_root = args.N2W_dataset_root

    # Initialize lists to store paths and data
    syncWideFovImage_paths = []
    narrowFovImg_paths = []
    narrowFovLabelXmlDirs = []
    dict_json = []

    # Loop through each viewpoint path and collect image and XML paths
    viewPoint_paths = glob.glob(N2W_dataset_root+r"\*")
    for viewPoint_path in viewPoint_paths:
        if  not Path.is_dir(Path(viewPoint_path)):
            continue
        syncWideFovImage_paths.append(glob.glob(viewPoint_path + r"\images_wide\*.jpg"))
        narrowFovImg_paths.append(glob.glob(viewPoint_path + r"\images_narrow\*.jpg"))
        narrowFovLabelXmlDirs.append(viewPoint_path+ r"\labels_narrow")

        # Load homography data from JSON files
        homographyJSONFolder = viewPoint_path + r"\homographies"
        homographyJSONPath = os.listdir(homographyJSONFolder)
        json_fp = open(homographyJSONFolder + "/" + homographyJSONPath[0])
        dict_json.append(json.load(json_fp))
        json_fp.close()

    # Retrieve command line arguments
    RandomDataNum = args.RandomDataNum
    outputDir = args.dataset_output
    outputName = args.outputName
    isSaveData = args.isSaveData
    image_index = 0
    
    # Generate a specified number of random data images
    for generate_image_index in tqdm.tqdm(range(RandomDataNum)):
        T1 = time.perf_counter()
        all_objNames = None
        all_objXyxys = None
        res = None
        wideFovImage = cv2.imread(random.choice(random.choice(syncWideFovImage_paths)))
        for i in range(len(narrowFovImg_paths)):
            j = random.randint(0, len(narrowFovImg_paths[i])-1)
            narrowFovImage_path = narrowFovImg_paths[i][j]

            # Construct the path for the XML label file
            narrowFovLabelXmlPath =  narrowFovLabelXmlDirs[i] + "/" + Path(narrowFovImage_path).stem + ".xml"
            syncWideFovImage_path = syncWideFovImage_paths[i][j]

            # Read object names and bounding boxes from XML
            lowFovObjNames, lowFovObjXyxys = None, None
            narrowFovImage = cv2.imread(narrowFovImage_path)
            if(Path(narrowFovLabelXmlPath).exists()):
                lowFovObjNames, lowFovObjXyxys = getObjectsFromXml(narrowFovLabelXmlPath)

            # Read the synchronized wide field of view image
            syncWideFovImage = cv2.imread(syncWideFovImage_path)

            # Load the homography matrix for the current viewpoint
            M = np.array(dict_json[i]["M"]["data"])
            M = M.reshape([3, 3])

            # Apply Gaussian blur to the narrow field of view image
            cv2.GaussianBlur(narrowFovImage, (7, 7), 0, narrowFovImage)

            # Warp the narrow field of view image onto the wide field of view image
            narrowFovImage_warped = cv2.warpPerspective(narrowFovImage, M,
                                                    [int(wideFovImage.shape[1]), int(wideFovImage.shape[0])],None,cv2.INTER_AREA,cv2.BORDER_REFLECT)
            
            # Create a mask for the wide field of view image
            mask = np.zeros_like(wideFovImage)

            # Define the screen points of the narrow field of view camera
            points_lowFov_screen = np.array([[[0, 0], [0, narrowFovImage.shape[0]], [narrowFovImage.shape[1], narrowFovImage.shape[0]], [narrowFovImage.shape[1], 0]]])
            points_lowFov_screen = points_lowFov_screen.reshape(1, -1, 2).astype(np.float32)

            # Warp the screen points using the homography matrix
            points_lowFov_screen_warped = cv2.perspectiveTransform(points_lowFov_screen, M)
            points_lowFov_screen_warped = points_lowFov_screen_warped.astype(np.int32)

            # Remove objects that overlap with the screen points of the narrow field of view camera
            if all_objNames is not None:
                all_objNames,  all_objXyxys = getObjectsIsNotInPolyon(all_objNames, all_objXyxys, points_lowFov_screen_warped)

            # If XML file exists, warp the object bounding boxes and update their positions
            if(Path(narrowFovLabelXmlPath).exists()):
                lowFovObjXyxys_warped = xyxys_warp(M, lowFovObjXyxys)
                for lowFovObjXyxy in lowFovObjXyxys_warped:
                    box = [0, 0, 0, 0]
                    box[0] = lowFovObjXyxy[0]
                    box[1] = lowFovObjXyxy[1]
                    box[2] = lowFovObjXyxy[2] - lowFovObjXyxy[0]
                    box[3] = lowFovObjXyxy[3] - lowFovObjXyxy[1]

                    # Initialize and update the tracker
                    tracker = cv2.TrackerKCF_create(params)
                    tracker.init(narrowFovImage_warped, box)
                    tracker.update(narrowFovImage_warped)
                    status, coord = tracker.update(syncWideFovImage)
                    if status :
                        lowFovObjXyxy[0] = coord[0]
                        lowFovObjXyxy[1] = coord[1]
                        lowFovObjXyxy[2] = coord[0] + coord[2]
                        lowFovObjXyxy[3] = coord[1] + coord[3]

                # Combine object names and bounding boxes from all viewpoints
                if all_objNames is not None:
                    all_objNames = all_objNames + lowFovObjNames
                    all_objXyxys = all_objXyxys + lowFovObjXyxys_warped
                else:
                    all_objNames = lowFovObjNames
                    all_objXyxys = lowFovObjXyxys_warped

            # Set the mask for the screen points of the narrow field of view camera to white
            cv2.fillPoly(mask, points_lowFov_screen_warped, [255, 255, 255])

            # Combine the warped narrow field of view image with the wide field of view image
            if res is None:
                res = cv2.copyTo(syncWideFovImage, mask, wideFovImage.copy())
            else:
                res = cv2.copyTo(syncWideFovImage, mask, res.copy())
            
            # Create a copy of the result for display
            res_show = res.copy()
            if all_objNames is not None:
                for all_objName, all_objXyxy in zip(all_objNames, all_objXyxys):
                    cv2.rectangle(res_show,[all_objXyxy[0], all_objXyxy[1]], [all_objXyxy[2], all_objXyxy[3]], [255, 0, 0], 2)
                    cv2.putText(res_show, all_objName, [all_objXyxy[0], all_objXyxy[1]], 0, 1, [0,255,0], 2)
        # Display the result in real-time if required
        if viewResult:
            cv2.namedWindow("res",cv2.WINDOW_NORMAL)
            cv2.imshow("res", res_show)
            cv2.waitKey(1)
        
        # Save the result if required
        if isSaveData:
            # Ensure the output directories exist
            if(not Path(outputDir + "/images/").exists()):
                os.makedirs(outputDir + "/images/")

            img_output_path = outputDir + "/images/" + outputName+str(image_index)+".jpg"
            
            if(not Path(outputDir + "/labels/").exists()):
                os.makedirs(outputDir + "/labels/")
            xml_output_path = outputDir + "/labels/" + outputName+str(image_index)+".xml"

            xml_root = None
            W = res.shape[1]
            H = res.shape[0]
            if all_objNames is not None:
                for box, pred_phrase in zip(all_objXyxys, all_objNames):
                    box[0] = max(0, box[0])
                    box[1] = max(0, box[1])
                    box[2] = min(box[2], W)
                    box[3] = min(box[3], H)
                    if not xml_root:
                            xml_root = create_voc_xml(outputName+str(image_index)+".jpg", width=W, height=H, depth=3,
                                                    object_name=pred_phrase,
                                                    bbox=[int(box[0]), int(box[1]),
                                                            int(box[2]), int(box[3])])
                    else:
                        xml_root = add_object_xml(xml_root, pred_phrase, bbox=[int(box[0]), int(box[1]),
                                                            int(box[2]), int(box[3])])
                        
            # Save the XML file and the image
            if xml_root is not None:
                prettyXml(xml_root, '    ', '\n')
                tree = ET.ElementTree(xml_root)
                tree.write(xml_output_path)
                cv2.imwrite(img_output_path, res)

        T2 = time.perf_counter()
        image_index = image_index + 1