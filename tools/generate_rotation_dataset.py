import os
from PIL import Image
import argparse
import logging
from pathlib import Path
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported image extensions
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')


def create_rotation_folders(output_dir: str, degrees: range) -> None:
    """
    Create folders for each rotation degree in the output directory.
    
    Args:
        output_dir: Path to the output directory
        degrees: Range of degrees to create folders for
        
    Raises:
        OSError: If folder creation fails
    """
    logger.info(f"Creating {len(degrees)} rotation folders in {output_dir}")
    
    for degree in degrees:
        folder_path = os.path.join(output_dir, str(degree))
        try:
            os.makedirs(folder_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {folder_path}: {e}")
            raise
    
    logger.info("All rotation folders created successfully")


def is_supported_image(filename: str) -> bool:
    """
    Check if a file has a supported image extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if the file has a supported image extension
    """
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)


def rotate_and_save_images(input_dir: str, output_dir: str, degrees: range, 
                          background_color: tuple = (255, 255, 255, 255)) -> None:
    """
    Rotate images through specified degrees and save with a solid background.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory where rotated images will be saved
        degrees: Range of degrees to rotate images through
        background_color: RGBA color for the background (default: white)
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
        ValueError: If no images are found in input directory
    """
    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if is_supported_image(f)]
    
    if not image_files:
        raise ValueError(f"No supported images found in '{input_dir}'. "
                        f"Supported extensions: {SUPPORTED_EXTENSIONS}")
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_name in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_name)
        
        try:
            with Image.open(image_path) as img:
                # Ensure the image has an alpha channel (transparency)
                img = img.convert("RGBA")
                
                # Rotate the image through each degree
                for degree in degrees:
                    # Rotate image
                    rotated_img = img.rotate(degree, expand=True)
                    
                    # Create background image with the same size as the rotated image
                    background = Image.new("RGBA", rotated_img.size, background_color)
                    
                    # Composite the rotated image over the background
                    combined = Image.alpha_composite(background, rotated_img)
                    
                    # Convert back to RGB (remove alpha channel) and save
                    final_image = combined.convert("RGB")
                    
                    # Save with original filename in degree folder
                    output_path = os.path.join(output_dir, str(degree), image_name)
                    final_image.save(output_path)
                
                logger.info(f"Processed image {i}/{len(image_files)}: {image_name}")
                
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {e}")
            continue
    
    logger.info(f"Completed processing {len(image_files)} images. "
               f"Rotated each through {len(degrees)} degrees.")


def main():
    """Main function to run the image rotation dataset generator."""
    parser = argparse.ArgumentParser(
        description='Generate a dataset of rotated images with solid backgrounds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir', '-i', 
        default="./white_background/val",
        help='Directory containing input images'
    )
    
    parser.add_argument(
        '--output-dir', '-o', 
        default="./dataset/val",
        help='Directory where rotated images will be saved'
    )
    
    parser.add_argument(
        '--start-degree', 
        type=int, 
        default=1,
        help='Starting rotation degree (inclusive)'
    )
    
    parser.add_argument(
        '--end-degree', 
        type=int, 
        default=360,
        help='Ending rotation degree (exclusive)'
    )
    
    parser.add_argument(
        '--step', 
        type=int, 
        default=1,
        help='Step size between rotation degrees'
    )
    
    parser.add_argument(
        '--background-color', 
        type=str, 
        default="white",
        choices=['white', 'black', 'transparent'],
        help='Background color for rotated images'
    )
    
    parser.add_argument(
        '--skip-folder-creation', 
        action='store_true',
        help='Skip creation of rotation folders (assumes they already exist)'
    )
    
    args = parser.parse_args()
    
    # Map color names to RGBA values
    color_map = {
        'white': (255, 255, 255, 255),
        'black': (0, 0, 0, 255),
        'transparent': (0, 0, 0, 0)
    }
    
    background_color = color_map[args.background_color]
    degrees = range(args.start_degree, args.end_degree, args.step)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create rotation folders if not skipped
        if not args.skip_folder_creation:
            create_rotation_folders(args.output_dir, degrees)
        
        # Rotate and save images
        rotate_and_save_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            degrees=degrees,
            background_color=background_color
        )
        
        logger.info("Dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
