import os
import glob
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_path, class_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    lines = []
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        class_id = class_map[class_name]

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        x = (xmin + xmax) / (2.0 * image_width)
        y = (ymin + ymax) / (2.0 * image_height)
        w = (xmax - xmin) / float(image_width)
        h = (ymax - ymin) / float(image_height)

        line = f"{class_id} {x} {y} {w} {h}"
        lines.append(line)

    return lines

# Define your class mapping (class names to integer IDs)
class_map = {
    'coal_rock': 0,
    'mithril_rock': 1,
    'adamantite_rock': 2,
    'iron_rock': 3,
    # Add more classes as needed
}

# Path to the directory containing the XML annotations
xml_dir = 'C:/Users/Elias/Desktop/Projects/Object Detection/TensorFlow/workspace/training_demo/images/all xml'

# Path to the directory where you want to save the YOLO-formatted annotations
yolo_dir = 'C:/Users/Elias/Desktop/Projects/Object Detection/yolov8/dataset/labels'

os.makedirs(yolo_dir, exist_ok=True)

# Convert XML annotations to YOLO format for each XML file
xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
for xml_file in xml_files:
    image_name = os.path.splitext(os.path.basename(xml_file))[0]
    yolo_file = os.path.join(yolo_dir, f"{image_name}.txt")

    lines = convert_xml_to_yolo(xml_file, class_map)

    with open(yolo_file, 'w') as f:
        f.write('\n'.join(lines))
