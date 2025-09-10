import os
import ast
import cv2
import numpy as np
import math
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_literal_eval(value: str) -> Tuple:
    """
    Safely evaluate a string representation of a tuple, handling edge cases.
    
    Args:
        value: String representation of a tuple
        
    Returns:
        Tuple: Evaluated tuple
        
    Raises:
        ValueError: If the string cannot be safely evaluated
    """
    try:
        # Handle empty values and edge cases
        value = value.strip()
        if not value or value == "()":
            return (0, 0)
        
        # Replace common issues in tuple formatting
        value = value.replace(",,", ",'',")
        value = value.replace("(", "").replace(")", "")  # Remove parentheses if present
        
        # Split and convert to numbers
        parts = value.split(',')
        if len(parts) != 2:
            raise ValueError(f"Expected 2 values, got {len(parts)}")
        
        x = float(parts[0]) if parts[0].strip() else 0.0
        y = float(parts[1]) if parts[1].strip() else 0.0
        
        return (x, y)
        
    except Exception as e:
        logger.error(f"Error parsing value '{value}': {e}")
        raise ValueError(f"Could not parse tuple from: {value}")


def load_points_from_file(points_file: str) -> List[Tuple[str, List[Tuple]]]:
    """
    Load points data from a text file.
    
    Args:
        points_file: Path to the points file
        
    Returns:
        List of tuples containing (image_path, [point1, point2])
        
    Raises:
        FileNotFoundError: If points file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"Points file '{points_file}' not found")
    
    points = []
    
    try:
        with open(points_file, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
                
            parts = line.split(';')
            if len(parts) < 3:
                logger.warning(f"Skipping line {line_num}: expected 3 parts, got {len(parts)}")
                continue
                
            image_path = parts[0].strip()
            point1_str = parts[1].strip()
            point2_str = parts[2].strip()
            
            try:
                point1 = safe_literal_eval(point1_str)
                point2 = safe_literal_eval(point2_str)
                points.append((image_path, [point1, point2]))
                
            except ValueError as e:
                logger.warning(f"Skipping line {line_num}: {e}")
                continue
                
    except Exception as e:
        raise ValueError(f"Error reading points file: {e}")
    
    logger.info(f"Loaded {len(points)} point pairs from {points_file}")
    return points


def rotate_image_to_horizontal(image: np.ndarray, point1: Tuple, point2: Tuple) -> np.ndarray:
    """
    Rotate an image so that the line between two points becomes horizontal.
    
    Args:
        image: Input image as numpy array
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Rotated image as numpy array
    """
    # Calculate the angle between the two points
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    
    logger.debug(f"Rotation angle: {angle_deg:.2f} degrees")
    
    # Get the center of the image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Calculate new bounding dimensions to avoid cropping
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform the actual rotation
    rotated_image = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # Black border
    )
    
    return rotated_image


def process_images_with_points(points_data: List[Tuple[str, List[Tuple]]], 
                              input_dir: str, 
                              output_dir: str,
                              display: bool = False) -> None:
    """
    Process all images using their corresponding points for rotation.
    
    Args:
        points_data: List of (image_path, [point1, point2]) tuples
        input_dir: Base directory for input images
        output_dir: Directory to save rotated images
        display: Whether to display images during processing
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    
    for image_rel_path, points in points_data:
        # Construct full image path
        image_path = os.path.join(input_dir, image_rel_path)
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
            
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                continue
            
            # Rotate image
            rotated_image = rotate_image_to_horizontal(image, points[0], points[1])
            
            # Save rotated image
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, rotated_image)
            
            # Display if requested
            if display:
                display_img = cv2.resize(rotated_image, (640, 480))
                cv2.imshow(f"Rotated: {image_name}", display_img)
                cv2.waitKey(500)  # Show for 500ms
                cv2.destroyAllWindows()
            
            processed_count += 1
            logger.info(f"Processed {processed_count}/{len(points_data)}: {image_name}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info(f"Successfully processed {processed_count} images")


def batch_rotate_images(input_dir: str, output_dir: str, rotation_code: int) -> None:
    """
    Batch rotate all images in a directory by a fixed rotation.
    
    Args:
        input_dir: Directory containing images to rotate
        output_dir: Directory to save rotated images
        rotation_code: OpenCV rotation code (e.g., cv2.ROTATE_90_CLOCKWISE)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found")
    
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    
    rotation_codes = {
        0: cv2.ROTATE_90_CLOCKWISE,
        1: cv2.ROTATE_180,
        2: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    
    if rotation_code not in rotation_codes:
        raise ValueError(f"Invalid rotation code. Use: {list(rotation_codes.keys())}")
    
    cv_code = rotation_codes[rotation_code]
    rotation_name = ["90° clockwise", "180°", "90° counterclockwise"][rotation_code]
    
    logger.info(f"Batch rotating images {rotation_name}")
    
    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image_path = os.path.join(input_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is not None:
                    rotated_image = cv2.rotate(image, cv_code)
                    output_path = os.path.join(output_dir, image_file)
                    cv2.imwrite(output_path, rotated_image)
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
    
    logger.info(f"Batch rotated {processed_count} images")


def main():
    """Main function to run the image alignment tool."""
    parser = argparse.ArgumentParser(
        description='Align images based on points or apply batch rotations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--align', '-a', action='store_true',
                          help='Align images using points from file')
    mode_group.add_argument('--batch-rotate', '-b', type=int, choices=[0, 1, 2],
                          help='Batch rotate images (0=90°CW, 1=180°, 2=90°CCW)')
    
    # Common arguments
    parser.add_argument('--input-dir', '-i', default='../pyqt/datasets/dataset/images/train',
                      help='Input directory containing images')
    parser.add_argument('--output-dir', '-o', default='./rotated',
                      help='Output directory for processed images')
    
    # Alignment-specific arguments
    parser.add_argument('--points-file', '-p', default='points.txt',
                      help='File containing points for alignment')
    parser.add_argument('--display', '-d', action='store_true',
                      help='Display images during processing')
    
    args = parser.parse_args()
    
    try:
        if args.align:
            # Alignment mode
            points_data = load_points_from_file(args.points_file)
            process_images_with_points(
                points_data, 
                args.input_dir, 
                args.output_dir,
                args.display
            )
            
        elif args.batch_rotate is not None:
            # Batch rotation mode
            batch_rotate_images(
                args.input_dir,
                args.output_dir,
                args.batch_rotate
            )
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    logger.info("Processing completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
