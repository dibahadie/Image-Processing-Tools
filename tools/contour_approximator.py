import os
import ast
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional, Dict, Any
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
KEYPOINT_COLOR = (0, 255, 0)  # Green
DEFAULT_DIAMETER = 15
DEFAULT_IMAGE_SIZE = (500, 500)


class KeypointVisualizer:
    """
    A class for visualizing and processing keypoints and contours from images.
    """
    
    def __init__(self, dataset_file: str = 'dataset.txt'):
        """
        Initialize the keypoint visualizer.
        
        Args:
            dataset_file: Path to the dataset file containing keypoint information
        """
        self.dataset_file = dataset_file
        self.contours_cache = None
        
    def show_image(self, image: np.ndarray, window_name: str = 'image') -> None:
        """
        Display an image in a window.
        
        Args:
            image: Image to display
            window_name: Name of the window
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_image_shape(self, image_path: str) -> Tuple[int, int, int]:
        """
        Get the shape of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (height, width, channels)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be read
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from '{image_path}'")
        
        return image.shape
    
    def visualize_keypoints(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray, 
        color: Tuple[int, int, int] = KEYPOINT_COLOR, 
        diameter: int = DEFAULT_DIAMETER,
        window_name: str = 'keypoints',
        resize: bool = True
    ) -> None:
        """
        Visualize keypoints on an image.
        
        Args:
            image: Input image
            keypoints: Array of keypoints (N, 2)
            color: Color for keypoints (B, G, R)
            diameter: Diameter of keypoint circles
            window_name: Name of the window
            resize: Whether to resize the image for display
        """
        if image is None:
            logger.error("Cannot visualize keypoints: image is None")
            return
            
        image = image.copy()
        
        # Draw keypoints
        for (x, y) in keypoints:
            cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        
        # Resize for display if requested
        if resize:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
        
        self.show_image(image, window_name)
    
    def safe_literal_eval(self, value: str) -> np.ndarray:
        """
        Safely evaluate a string representation of a list/array.
        
        Args:
            value: String to evaluate
            
        Returns:
            numpy.ndarray: Evaluated array
            
        Raises:
            ValueError: If evaluation fails
        """
        try:
            # Handle edge cases in the data format
            value = value.strip()
            value = value.replace(",,", ",0,")  # Replace empty values with 0
            value = value.replace("'", "")  # Remove quotes if present
            
            # Evaluate and convert to numpy array
            result = ast.literal_eval(value)
            return np.array(result, dtype=np.float32)
            
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error evaluating string '{value}': {e}")
            raise ValueError(f"Could not parse value: {value}")
    
    def load_all_contours(self) -> List[Tuple[str, np.ndarray]]:
        """
        Load all contours from the dataset file.
        
        Returns:
            List of tuples containing (image_name, contour_points)
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(f"Dataset file '{self.dataset_file}' not found")
        
        contours = []
        
        try:
            with open(self.dataset_file, 'r') as file:
                lines = file.readlines()
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split(';')
                if len(parts) < 2:
                    logger.warning(f"Skipping line {line_num}: expected at least 2 parts, got {len(parts)}")
                    continue
                
                image_name = parts[0].strip()
                contour_data = parts[1].strip()
                
                try:
                    contour_points = self.safe_literal_eval(contour_data)
                    contours.append((image_name, contour_points))
                    
                except ValueError as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading dataset file: {e}")
            raise
            
        logger.info(f"Loaded {len(contours)} contours from {self.dataset_file}")
        return contours
    
    def get_keypoints(
        self, 
        image_name: str, 
        contours: List[Tuple[str, np.ndarray]],
        target_size: int = 299,
        padding: int = 10
    ) -> Optional[np.ndarray]:
        """
        Get keypoints for a specific image.
        
        Args:
            image_name: Name of the image to find
            contours: List of contours from load_all_contours()
            target_size: Size to scale keypoints to
            padding: Padding to add to keypoints
            
        Returns:
            Array of keypoints or None if not found
        """
        for contour in contours:
            if image_name == contour[0]:
                try:
                    # Normalize and scale keypoints
                    scaler = MinMaxScaler()
                    kps = scaler.fit_transform(contour[1])
                    kps = np.array(kps)
                    kps = kps * target_size
                    kps = np.int32(kps)
                    kps += padding
                    return kps
                    
                except Exception as e:
                    logger.error(f"Error processing keypoints for {image_name}: {e}")
                    return None
        
        logger.warning(f"No keypoints found for image: {image_name}")
        return None
    
    def draw_contours_on_blank(
        self, 
        contours: np.ndarray, 
        image_size: Tuple[int, int] = (400, 400),
        background_value: int = 10,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw contours on a blank image.
        
        Args:
            contours: Contour points to draw
            image_size: Size of the output image
            background_value: Value for the background
            color: Color for the contours
            thickness: Thickness of contour lines
            
        Returns:
            Image with drawn contours
        """
        blank = np.zeros(image_size, dtype='uint8')
        blank = blank + background_value
        
        # Convert to 3-channel for color drawing if needed
        if len(color) == 3:
            blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        
        cv2.drawContours(blank, [contours], -1, color, thickness)
        return blank
    
    def approximate_contour(
        self, 
        contour: np.ndarray, 
        epsilon_factor: float = 0.01
    ) -> np.ndarray:
        """
        Approximate a contour using Douglas-Peucker algorithm.
        
        Args:
            contour: Input contour points
            epsilon_factor: Factor for approximation accuracy
            
        Returns:
            Approximated contour
        """
        if contour is None or len(contour) < 3:
            return contour
            
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        return approx
    
    def test_contour_approximation(
        self, 
        image_path: str, 
        contours_data: List[Tuple[str, np.ndarray]],
        epsilon_range: Tuple[float, float] = (0.001, 0.05),
        num_tests: int = 30
    ) -> None:
        """
        Test contour approximation with different epsilon values.
        
        Args:
            image_path: Path to the image
            contours_data: Loaded contours data
            epsilon_range: Range of epsilon values to test
            num_tests: Number of tests to perform
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file '{image_path}' not found")
            return
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image from '{image_path}'")
            return
            
        # Extract image name from path
        image_name = os.path.basename(image_path)
        
        # Get keypoints for this image
        contour = self.get_keypoints(image_name, contours_data)
        if contour is None:
            logger.error(f"No keypoints found for {image_name}")
            return
            
        logger.info(f"Testing contour approximation for {image_name}")
        logger.info(f"Original contour has {len(contour)} points")
        
        # Test different epsilon values
        for eps in np.linspace(epsilon_range[0], epsilon_range[1], num_tests):
            approx = self.approximate_contour(contour, eps)
            text = f"eps={eps:.4f}, num_pts={len(approx)}"
            
            # Visualize the approximated contour
            self.visualize_keypoints(
                image, 
                approx, 
                color=KEYPOINT_COLOR, 
                diameter=DEFAULT_DIAMETER,
                window_name=f"Approximation: {text}",
                resize=True
            )
            
            print(text)


def main():
    """Main function to demonstrate keypoint visualization functionality."""
    parser = argparse.ArgumentParser(
        description='Visualize and process keypoints from images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset-file', '-d', default='dataset.txt',
                       help='File containing keypoint data')
    parser.add_argument('--image-dir', '-i', default='./tests',
                       help='Directory containing test images')
    parser.add_argument('--test-approximation', '-t', action='store_true',
                       help='Test contour approximation with different epsilon values')
    parser.add_argument('--image-name', '-n', default='20240714_124442.jpg',
                       help='Specific image name to process')
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = KeypointVisualizer(dataset_file=args.dataset_file)
        
        # Load all contours
        contours = visualizer.load_all_contours()
        
        if args.test_approximation:
            # Test contour approximation
            test_image_path = os.path.join(args.image_dir, '1', args.image_name)
            visualizer.test_contour_approximation(test_image_path, contours)
            
        else:
            # Example: Process specific images
            test_images = [
                './tests/2/20240714_124709_rotate.jpg',
                './tests/2/20240714_124822_rotate.jpg',
                './tests/2/20240714_141631_rotate.jpg'
            ]
            
            for image_path in test_images:
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image_name = os.path.basename(image_path)
                    
                    keypoints = visualizer.get_keypoints(image_name, contours)
                    if keypoints is not None:
                        visualizer.visualize_keypoints(image, keypoints)
                    else:
                        logger.warning(f"No keypoints found for {image_name}")
                else:
                    logger.warning(f"Image not found: {image_path}")
                    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
