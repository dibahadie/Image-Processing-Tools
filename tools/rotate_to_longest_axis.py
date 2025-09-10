import math
import cv2
import numpy as np
from ultralytics import YOLO
import imutils
import argparse
import os


def predict_keypoints(image, model_path='./best.pt'):
    """
    Predict keypoints in an image using a YOLO model.
    
    Args:
        image: Input image as numpy array
        model_path: Path to the YOLO model weights file
        
    Returns:
        numpy.ndarray: Array of keypoints (x, y coordinates)
        
    Raises:
        RuntimeError: If no masks are detected in the image
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Perform prediction
    results = model(image)
    
    # Check if masks are detected
    if results[0].masks is None or len(results[0].masks.xy) == 0:
        raise RuntimeError("No masks detected in the image")
    
    # Extract keypoints from the first mask
    keypoints = results[0].masks.xy[0]
    
    return keypoints


def calculate_angle(point1, point2):
    """
    Calculate the angle between two points relative to the horizontal axis.
    
    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)
        
    Returns:
        float: Angle in degrees
    """
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    return math.degrees(np.arctan2(delta_y, delta_x))


def calculate_bounding_box(image, model_path='./best.pt'):
    """
    Calculate the minimum area rectangle bounding box around detected keypoints.
    
    Args:
        image: Input image as numpy array
        model_path: Path to the YOLO model weights file
        
    Returns:
        numpy.ndarray: Array of four points representing the bounding box
    """
    # Get keypoints from the model
    keypoints = predict_keypoints(image, model_path)
    
    # Calculate minimum area rectangle
    rect = cv2.minAreaRect(keypoints)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    return box


def rotate_image(image, model_path='./best.pt'):
    """
    Rotate image to align the longest side of the detected object with the horizontal axis.
    
    Args:
        image: Input image as numpy array
        model_path: Path to the YOLO model weights file
        
    Returns:
        numpy.ndarray: Rotated image
    """
    # Get bounding box
    bounding_box = calculate_bounding_box(image, model_path)
    
    # Calculate the lengths of the sides
    side_lengths = [np.linalg.norm(bounding_box[i] - bounding_box[i + 1]) 
                   for i in range(len(bounding_box) - 1)]
    
    # Add the length between the last and first point
    side_lengths.append(np.linalg.norm(bounding_box[3] - bounding_box[0]))
    
    # Find the indices of the largest side
    max_index = np.argmax(side_lengths)
    
    # Find the two points that form the largest side
    if max_index == 3:  # Connection between last and first point
        p1, p2 = bounding_box[3], bounding_box[0]
    else:
        p1, p2 = bounding_box[max_index], bounding_box[max_index + 1]
    
    # Calculate the angle between this side and the horizontal axis
    angle = calculate_angle(p1, p2)
    print(f"Rotation angle: {angle:.2f} degrees")
    
    # Rotate the image to align the largest side with the horizontal axis
    rotated_image = imutils.rotate(image, angle)
    
    return rotated_image, bounding_box, angle


def main():
    """Main function to demonstrate the image rotation functionality."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Rotate image based on detected object orientation')
    parser.add_argument('--input', '-i', default='./img.jpg', 
                       help='Input image path (default: ./img.jpg)')
    parser.add_argument('--output', '-o', default='./output.jpg', 
                       help='Output image path (default: ./output.jpg)')
    parser.add_argument('--model', '-m', default='./best.pt', 
                       help='Path to YOLO model weights (default: ./best.pt)')
    parser.add_argument('--display', '-d', action='store_true',
                       help='Display the result in a window')
    parser.add_argument('--draw-box', '-b', action='store_true',
                       help='Draw the bounding box on the output image')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    # Check if model file exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    try:
        # Read input image
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Could not read image from '{args.input}'.")
            return
        
        print(f"Processing image: {args.input}")
        
        # Rotate the image
        rotated_img, bounding_box, angle = rotate_image(img, args.model)
        
        # Draw bounding box if requested
        if args.draw_box:
            cv2.drawContours(rotated_img, [bounding_box], 0, (0, 0, 255), 2)
        
        # Resize for display (if needed)
        display_img = cv2.resize(rotated_img, (640, 480))
        
        # Save the result
        cv2.imwrite(args.output, rotated_img)
        print(f"Result saved to: {args.output}")
        
        # Display the result if requested
        if args.display:
            cv2.imshow('Aligned Image', display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
