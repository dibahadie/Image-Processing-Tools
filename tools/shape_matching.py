import cv2
import numpy as np
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
DEFAULT_DISPLAY_SIZE = (500, 500)


class ShapeMatcher:
    """
    A class for comparing and matching shapes using contour analysis.
    """
    
    def __init__(self, dataset_file: str = 'dataset.txt'):
        """
        Initialize the shape matcher.
        
        Args:
            dataset_file: Path to the dataset file containing contour information
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
        if image is None:
            logger.warning("Cannot display None image")
            return
            
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def visualize_keypoints(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray, 
        color: Tuple[int, int, int] = KEYPOINT_COLOR, 
        diameter: int = DEFAULT_DIAMETER,
        window_name: str = 'keypoints'
    ) -> None:
        """
        Visualize keypoints on an image.
        
        Args:
            image: Input image
            keypoints: Array of keypoints (N, 2)
            color: Color for keypoints (B, G, R)
            diameter: Diameter of keypoint circles
            window_name: Name of the window
        """
        if image is None or keypoints is None:
            logger.warning("Cannot visualize keypoints: image or keypoints is None")
            return
            
        image_copy = image.copy()
        
        # Draw keypoints
        for (x, y) in keypoints:
            cv2.circle(image_copy, (int(x), int(y)), diameter, color, -1)
        
        # Resize for display
        image_copy = cv2.resize(image_copy, DEFAULT_DISPLAY_SIZE)
        
        self.show_image(image_copy, window_name)
    
    def load_contours_from_file(self) -> List[Tuple[str, np.ndarray]]:
        """
        Load contours from a dataset file.
        
        Returns:
            List of tuples containing (image_name, contour_points)
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
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
                contour_data = ';'.join(parts[1:]).strip()  # Handle multiple semicolons
                
                try:
                    # Parse contour data (assuming it's in a readable format)
                    # This is a placeholder - you'll need to implement your actual parsing logic
                    contour_points = self.parse_contour_data(contour_data)
                    contours.append((image_name, contour_points))
                    
                except Exception as e:
                    logger.warning(f"Skipping line {line_num}: error parsing contour data: {e}")
                    continue
                    
        except FileNotFoundError:
            logger.error(f"Dataset file '{self.dataset_file}' not found")
            raise
        except Exception as e:
            logger.error(f"Error reading dataset file: {e}")
            raise
            
        logger.info(f"Loaded {len(contours)} contours from {self.dataset_file}")
        return contours
    
    def parse_contour_data(self, contour_data: str) -> np.ndarray:
        """
        Parse contour data from string format.
        This is a placeholder - implement your actual parsing logic here.
        
        Args:
            contour_data: String representation of contour data
            
        Returns:
            numpy.ndarray: Parsed contour points
        """
        # Implement your specific parsing logic here
        # This is just a placeholder example
        try:
            # Example parsing - adjust based on your actual data format
            points = []
            for point_str in contour_data.split(','):
                if point_str.strip():
                    x, y = map(float, point_str.strip('()').split(','))
                    points.append([x, y])
            return np.array(points, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Error parsing contour data: {e}")
    
    def get_contour_by_name(self, image_name: str, contours: List[Tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
        """
        Get contour for a specific image name.
        
        Args:
            image_name: Name of the image to find
            contours: List of contours from load_contours_from_file()
            
        Returns:
            Contour points or None if not found
        """
        for contour in contours:
            if image_name == contour[0]:
                return contour[1]
        
        logger.warning(f"No contour found for image: {image_name}")
        return None
    
    def predict_contour(self, image: np.ndarray, model_path: str) -> Optional[np.ndarray]:
        """
        Predict contour using YOLO model.
        
        Args:
            image: Input image
            model_path: Path to YOLO model weights
            
        Returns:
            Predicted contour points or None if prediction fails
        """
        try:
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(model_path)

            # Perform prediction
            results = model(image, task='segment')
            
            if (results is None or len(results) == 0 or 
                results[0].masks is None or len(results[0].masks) == 0):
                logger.warning("No segmentation mask found in prediction results")
                return None

            contour_points = results[0].masks[0].xy[0]
            return np.int32(contour_points)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def draw_contour_mask(
        self, 
        image: np.ndarray, 
        contour: np.ndarray,
        thickness: int = 5
    ) -> np.ndarray:
        """
        Draw contour as a binary mask.
        
        Args:
            image: Reference image for size
            contour: Contour points to draw
            thickness: Thickness of contour lines
            
        Returns:
            Binary mask with drawn contour
        """
        if image is None or contour is None:
            logger.warning("Cannot draw contour: image or contour is None")
            return None
            
        # Create blank mask with same dimensions as input image
        blank = np.zeros(image.shape[:2], dtype='uint8')
        
        # Draw contour
        cv2.drawContours(blank, [contour], -1, 255, thickness)
        
        return blank
    
    def compare_contours(
        self, 
        contour1: np.ndarray, 
        contour2: np.ndarray, 
        method: int = cv2.CONTOURS_MATCH_I1
    ) -> float:
        """
        Compare two contours using shape matching.
        
        Args:
            contour1: First contour
            contour2: Second contour
            method: OpenCV contour matching method
            
        Returns:
            Match value (lower means better match)
        """
        if contour1 is None or contour2 is None:
            logger.warning("Cannot compare contours: one or both are None")
            return float('inf')
            
        try:
            return cv2.matchShapes(contour1, contour2, method, 0.0)
        except Exception as e:
            logger.error(f"Error comparing contours: {e}")
            return float('inf')
    
    def find_best_match(
        self, 
        query_contour: np.ndarray, 
        database_contours: List[Tuple[str, np.ndarray]],
        comparison_method: int = cv2.CONTOURS_MATCH_I1
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching contour in the database.
        
        Args:
            query_contour: Contour to match
            database_contours: List of contours from database
            comparison_method: OpenCV contour matching method
            
        Returns:
            Tuple of (best_match_image_name, best_match_value)
        """
        if query_contour is None:
            logger.warning("Cannot find match: query contour is None")
            return None, float('inf')
            
        best_match_value = float('inf')
        best_match_name = None
        
        for image_name, db_contour in database_contours:
            if db_contour is not None:
                match_value = self.compare_contours(query_contour, db_contour, comparison_method)
                
                if match_value < best_match_value:
                    best_match_value = match_value
                    best_match_name = image_name
        
        return best_match_name, best_match_value
    
    def process_image_comparison(
        self, 
        query_image_path: str, 
        model_path: str,
        reference_image_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process image comparison against database.
        
        Args:
            query_image_path: Path to query image
            model_path: Path to YOLO model
            reference_image_name: Optional specific image to compare against
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        try:
            # Load query image
            query_image = cv2.imread(query_image_path)
            if query_image is None:
                raise ValueError(f"Could not read query image: {query_image_path}")
            
            # Predict contour from query image
            predicted_contour = self.predict_contour(query_image, model_path)
            if predicted_contour is None:
                raise ValueError("Failed to predict contour from query image")
            
            results['predicted_contour'] = predicted_contour
            
            # Load database contours
            database_contours = self.load_contours_from_file()
            
            # Find best match in database
            best_match_name, best_match_value = self.find_best_match(
                predicted_contour, database_contours
            )
            
            results['best_match'] = {
                'image_name': best_match_name,
                'match_value': best_match_value
            }
            
            # Compare with specific reference image if provided
            if reference_image_name:
                reference_contour = self.get_contour_by_name(reference_image_name, database_contours)
                if reference_contour is not None:
                    reference_match = self.compare_contours(predicted_contour, reference_contour)
                    results['reference_match'] = {
                        'image_name': reference_image_name,
                        'match_value': reference_match
                    }
            
            logger.info(f"Best match: {best_match_name} with value: {best_match_value:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing image comparison: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Main function to demonstrate shape matching functionality."""
    parser = argparse.ArgumentParser(
        description='Compare shapes using contour matching.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--query-image', '-q', required=True,
                       help='Path to query image for comparison')
    parser.add_argument('--model-path', '-m', default='./best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--dataset-file', '-d', default='dataset.txt',
                       help='File containing contour database')
    parser.add_argument('--reference-image', '-r',
                       help='Specific reference image name to compare against')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Visualize results')
    
    args = parser.parse_args()
    
    try:
        # Initialize shape matcher
        matcher = ShapeMatcher(dataset_file=args.dataset_file)
        
        # Process comparison
        results = matcher.process_image_comparison(
            args.query_image, 
            args.model_path,
            args.reference_image
        )
        
        if 'error' in results:
            logger.error(f"Comparison failed: {results['error']}")
            return 1
        
        # Print results
        print(f"Best match: {results['best_match']['image_name']}")
        print(f"Match value: {results['best_match']['match_value']:.6f}")
        
        if 'reference_match' in results:
            print(f"Reference match: {results['reference_match']['image_name']}")
            print(f"Reference value: {results['reference_match']['match_value']:.6f}")
        
        # Visualize if requested
        if args.visualize:
            query_image = cv2.imread(args.query_image)
            matcher.visualize_keypoints(query_image, results['predicted_contour'])
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
