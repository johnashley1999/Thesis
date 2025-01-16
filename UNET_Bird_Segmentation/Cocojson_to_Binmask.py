import json
import numpy as np
import cv2
import os
from PIL import Image

def create_mask_from_polygon(polygon, height, width):
    """Create a binary mask from polygon vertices."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], 255)
    return mask

def convert_coco_to_masks(coco_data, directory):
    """Convert COCO JSON annotations to binary masks."""
    # Create lookup dictionary for images
    image_lookup = {img['id']: img for img in coco_data['images']}
    
    # Track which images have been processed
    processed_images = set()
    
    # Process each annotation
    for ann in coco_data['annotations']:
        # Get image info
        img_info = image_lookup[ann['image_id']]
        if img_info['file_name'] in processed_images:
            continue
            
        height = img_info['height']
        width = img_info['width']
        
        # Create mask from all annotations for this image
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get all annotations for this image
        image_annotations = [a for a in coco_data['annotations'] if a['image_id'] == ann['image_id']]
        
        for img_ann in image_annotations:
            if isinstance(img_ann['segmentation'], list) and len(img_ann['segmentation']) > 0:
                for segment in img_ann['segmentation']:
                    polygon = np.array(segment).reshape(-1, 2)
                    temp_mask = create_mask_from_polygon(polygon, height, width)
                    mask = cv2.bitwise_or(mask, temp_mask)
        
        # Create mask filename
        base_name = os.path.splitext(img_info['file_name'])[0]
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(directory, mask_filename)
        
        # Save the mask
        try:
            Image.fromarray(mask).save(mask_path)
            print(f"Created mask for image: {img_info['file_name']}")
            processed_images.add(img_info['file_name'])
        except Exception as e:
            print(f"Error saving mask for {img_info['file_name']}: {str(e)}")

def main():
    # Get the directory containing the script
    base_directory = os.path.dirname(os.path.abspath(__file__))
    directory_mods = ["Data/train", "Data/test", "Data/valid"]
    
    for directory_mod in directory_mods:
        current_directory = os.path.join(base_directory, directory_mod)
        # Path to COCO JSON file
        json_path = os.path.join(current_directory, '_annotations.coco.json') 
        print(f"Looking for annotations file in: {json_path}")
        
        try:
            # Load COCO JSON
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            
            # Convert annotations to masks
            convert_coco_to_masks(coco_data, current_directory)
            print("Conversion completed successfully!")
            
        except FileNotFoundError:
            print(f"Could not find JSON file: {json_path}")
        except Exception as e:
            print(f"Error during conversion: {str(e)}")

if __name__ == '__main__':
    main()