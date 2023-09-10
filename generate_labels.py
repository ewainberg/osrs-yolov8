import os
from ultralytics import YOLO

model = YOLO('best.pt')
classNames = ['coal_rock', 'mithril_rock', 'adamantite_rock', 'iron_rock']

# Set the root directory path
root_dir = 'dataset/images'

# Set the subdirectory names
sub_dirs = ['train', 'val']

# Iterate through the subdirectories
for sub_dir in sub_dirs:
    folder_path = os.path.join(root_dir, sub_dir)
    
    # Iterate through the image files in the subdirectory
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            
            # Process each image file as needed
            model.predict(image_path, save_txt=True, imgsz=1024, conf=0.5, device="0")
