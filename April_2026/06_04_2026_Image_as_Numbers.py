"""
================================================================================
FILE: 06_04_2026_Image_as_Numbers.py
TITLE: Understanding Images as Numerical Matrices
================================================================================

This comprehensive educational script demonstrates how digital images are
represented and manipulated as numerical matrices (arrays). It creates synthetic
images using 3D NumPy arrays, visualizes them, analyzes their numerical
properties, and performs pixel-level modifications.

KEY CONCEPTS:
- Images as 3D matrices (height × width × color channels)
- RGB color representation (0-255 per channel)
- Channel extraction and analysis
- Image statistics (min, max, mean)
- Pixel manipulation and transformation
- Color operations (brighten, darken, gradient)

MATHEMATICAL FOUNDATIONS:
- Matrix indexing and slicing
- Broadcasting operations
- Statistical analysis on matrices
- Normalization and scaling

DEPENDENCIES:
- numpy: Numerical matrix operations
- matplotlib: Image visualization
- PIL: Image I/O operations

AUTHOR: Computer Vision Lab
DATE: April 6, 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# ============================================================================
# SECTION 1: FOUNDATIONAL CONCEPTS - COLOR AND IMAGES
# ============================================================================

"""
UNDERSTANDING DIGITAL IMAGES AS MATRICES
==========================================

What is a Digital Image?
------------------------
A digital image is a 2D grid of pixels, where each pixel represents a color.
Each pixel's color is defined by numeric values in specific color spaces.

COLOR REPRESENTATION: RGB (Red, Green, Blue)
---------------------------------------------
The RGB color model represents colors as combinations of three primary colors:
- Red (R): 0-255 intensity (0=no red, 255=full red)
- Green (G): 0-255 intensity (0=no green, 255=full green)
- Blue (B): 0-255 intensity (0=no blue, 255=full blue)

EXAMPLE RGB COLORS:
- (255, 0, 0) = pure red
- (0, 255, 0) = pure green
- (0, 0, 255) = pure blue
- (255, 255, 0) = red + green = yellow
- (255, 255, 255) = red + green + blue = white
- (0, 0, 0) = no light = black
- (128, 128, 128) = gray (equal amounts)

MATRIX REPRESENTATION
---------------------
An image of height H, width W is stored as a 3D array:
- Dimensions: [H, W, 3] where H=height, W=width, 3=RGB channels
- Element [i, j, 0] = Red channel value at pixel (i, j)
- Element [i, j, 1] = Green channel value at pixel (i, j)
- Element [i, j, 2] = Blue channel value at pixel (i, j)

MEMORY LAYOUT
-------------
Total pixels = H × W
Total matrix elements = H × W × 3
Each element is typically uint8 (0-255)
Memory = H × W × 3 × 1 byte = H × W × 3 bytes

EXAMPLE: 100×100×3 image
Total elements = 30,000
Memory = 30 KB
"""

# ============================================================================
# SECTION 2: CREATING AND DISPLAYING IMAGES
# ============================================================================

def create_solid_color_image(height=100, width=100, color=(255, 0, 0)):
    """
    Create a solid-colored image as a NumPy array.
    
    MATHEMATICAL EXPLANATION:
    We create a matrix of shape [height, width, 3] where every pixel
    has the same RGB values (color). This is done using:
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = color
    
    This uses NumPy broadcasting - assigning a single RGB tuple to all
    matrix positions. Broadcasting is efficient because:
    - It doesn't create intermediate copies
    - Memory access is optimized
    - The operation is vectorized (no Python loops)
    
    Args:
        height (int): Height in pixels
        width (int): Width in pixels
        color (tuple): RGB color as (R, G, B) tuple with values 0-255
        
    Returns:
        np.ndarray: Image array of shape [height, width, 3]
        
    IMPORTANT: We use dtype=uint8 because:
    - uint8 = unsigned 8-bit integer = 0-255 range
    - Standard for image format
    - Memory efficient (1 byte per channel per pixel)
    """
    # Create array of zeros with shape (height, width, 3)
    # All values initialized to 0 (black)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign color to every pixel using broadcasting
    # image[:, :, :] means all rows, all columns, all channels
    image[:, :, :] = color
    
    return image


def create_rgb_channel_images(height=100, width=100):
    """
    Create three separate images showing Red, Green, and Blue channels
    at full intensity.
    
    EXPLANATION:
    This demonstrates channel separation. We create three images where:
    - Red image: R=255, G=0, B=0 (pure red)
    - Green image: R=0, G=255, B=0 (pure green)
    - Blue image: R=0, G=0, B=255 (pure blue)
    
    Args:
        height (int): Image height
        width (int): Image width
        
    Returns:
        tuple: (red_image, green_image, blue_image) - three numpy arrays
    """
    # Pure red image - maximum red channel, no green or blue
    red_image = np.zeros((height, width, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # Set all red channel values to 255
    
    # Pure green image - maximum green channel, no red or blue
    green_image = np.zeros((height, width, 3), dtype=np.uint8)
    green_image[:, :, 1] = 255  # Set all green channel values to 255
    
    # Pure blue image - maximum blue channel, no red or green
    blue_image = np.zeros((height, width, 3), dtype=np.uint8)
    blue_image[:, :, 2] = 255  # Set all blue channel values to 255
    
    return red_image, green_image, blue_image


def extract_channels(image):
    """
    Extract individual color channels from an image.
    
    MATHEMATICAL OPERATION:
    Given image of shape [H, W, 3], we extract:
    - red = image[:, :, 0]     (shape: [H, W])
    - green = image[:, :, 1]   (shape: [H, W])
    - blue = image[:, :, 2]    (shape: [H, W])
    
    Each channel is now a 2D grayscale matrix where values represent
    the intensity of that color component.
    
    Args:
        image (np.ndarray): Image array of shape [H, W, 3]
        
    Returns:
        tuple: (red_channel, green_channel, blue_channel) - 2D arrays
    """
    # Extract each color channel using array indexing
    # The : means "all rows" and "all columns"
    red = image[:, :, 0]      # All pixels, red channel (index 0)
    green = image[:, :, 1]    # All pixels, green channel (index 1)
    blue = image[:, :, 2]     # All pixels, blue channel (index 2)
    
    return red, green, blue


def calculate_channel_statistics(channel):
    """
    Calculate statistical measures for a color channel.
    
    STATISTICAL CONCEPTS:
    - Min: Minimum value in the matrix = darkest pixel
    - Max: Maximum value in the matrix = brightest pixel
    - Mean: Average value = average brightness
    - Std: Standard deviation = how much values vary
    - Median: Middle value when sorted
    
    MATHEMATICAL FORMULAS:
    - Mean = (sum of all values) / (number of values)
    - Std = sqrt(mean of (value - mean)^2)
    - These are computed using efficient NumPy operations
    
    Args:
        channel (np.ndarray): 2D array representing one color channel
        
    Returns:
        dict: Dictionary with statistical measures
    """
    # Calculate statistics using NumPy functions
    # These are vectorized operations (no Python loops)
    statistics = {
        'min': np.min(channel),          # Minimum value
        'max': np.max(channel),          # Maximum value
        'mean': np.mean(channel),        # Average value
        'median': np.median(channel),    # Middle value
        'std': np.std(channel),          # Standard deviation
        'variance': np.var(channel),     # Variance (std squared)
    }
    
    return statistics


def visualize_image(image, title="Image"):
    """
    Display an image using matplotlib.
    
    VISUALIZATION DETAILS:
    - Uses matplotlib for interactive display
    - Handles image format automatically
    - Transpose not needed for NumPy arrays with channels last
    
    Args:
        image (np.ndarray): Image array of shape [H, W, 3]
        title (str): Title for the display window
    """
    # Create a figure and axis
    plt.figure(figsize=(8, 8))
    
    # Display image
    # imshow expects shape [H, W, 3] or [H, W]
    # Values should be uint8 (0-255) or float (0.0-1.0)
    plt.imshow(image)
    
    # Add title
    plt.title(title, fontsize=16)
    
    # Remove axis labels for cleaner display
    plt.axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 3: PIXEL MANIPULATION AND TRANSFORMATIONS
# ============================================================================

def brighten_image(image, factor=1.5):
    """
    Brighten an image by multiplying pixel values.
    
    MATHEMATICAL OPERATION:
    For each pixel: new_value = old_value × factor
    
    CONSIDERATIONS:
    - Multiplication increases all values proportionally
    - Values that exceed 255 are clipped to 255
    - Uses np.clip to prevent overflow
    
    EXAMPLE:
    If pixel = 200 and factor = 1.5:
    new_value = 200 × 1.5 = 300
    clipped = min(300, 255) = 255
    
    Args:
        image (np.ndarray): Input image
        factor (float): Brightness multiplier (>1 brightens, <1 darkens)
        
    Returns:
        np.ndarray: Brightened image
    """
    # Convert to float for computation to avoid overflow
    # float32 allows values >255 temporarily
    brightened = image.astype(np.float32) * factor
    
    # Clip values to valid range [0, 255]
    # np.clip(array, min, max) ensures all values are in range
    brightened = np.clip(brightened, 0, 255)
    
    # Convert back to uint8
    brightened = brightened.astype(np.uint8)
    
    return brightened


def darken_image(image, factor=0.5):
    """
    Darken an image by multiplying pixel values by a factor < 1.
    
    MATHEMATICAL OPERATION:
    For each pixel: new_value = old_value × factor (where 0 < factor < 1)
    
    This is equivalent to brightening with factor < 1.
    Uses the same mathematical principle as brighten_image.
    
    Args:
        image (np.ndarray): Input image
        factor (float): Darkness multiplier (0 to 1, less brightens)
        
    Returns:
        np.ndarray: Darkened image
    """
    # Use brighten_image with factor < 1 for darkening
    # This is more efficient than duplicating the code
    return brighten_image(image, factor)


def create_gradient_image(height=100, width=100):
    """
    Create an image with a gradient from black to white.
    
    MATHEMATICAL OPERATION:
    We create a gradient where pixel values increase linearly from left to right.
    
    For each pixel at column j (0 to width-1):
    value = (j / width) × 255
    
    This creates a smooth transition from black (0) to white (255).
    
    VISUAL RESULT:
    The image appears to gradually brighten from left to right.
    
    Args:
        height (int): Image height
        width (int): Image width
        
    Returns:
        np.ndarray: Gradient image
    """
    # Create empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient values for each column
    # np.linspace(0, 255, width) creates width values from 0 to 255
    gradient_values = np.linspace(0, 255, width, dtype=np.uint8)
    
    # Assign gradient to each row
    # This uses broadcasting: (1, width) broadcasts to (height, width)
    for row in range(height):
        image[row, :, :] = gradient_values[:, np.newaxis]
    
    return image


def add_noise_to_image(image, noise_level=0.1):
    """
    Add random noise to an image.
    
    MATHEMATICAL OPERATION:
    For each pixel: new_value = old_value + noise
    where noise is random in range [-noise_range, noise_range]
    
    This simulates real-world image degradation or creates artistic effects.
    
    Args:
        image (np.ndarray): Input image
        noise_level (float): Noise intensity (0-1 scale)
        
    Returns:
        np.ndarray: Image with added noise
    """
    # Create noise array with same shape as image
    noise_range = int(noise_level * 255)
    
    # Generate random noise
    # np.random.randint generates random integers in range
    noise = np.random.randint(-noise_range, noise_range, image.shape)
    
    # Add noise to image
    noisy = image.astype(np.int16) + noise
    
    # Clip to valid range
    noisy = np.clip(noisy, 0, 255)
    
    # Convert back to uint8
    noisy = noisy.astype(np.uint8)
    
    return noisy


# ============================================================================
# SECTION 4: ANALYSIS AND STATISTICS
# ============================================================================

def analyze_image(image):
    """
    Perform comprehensive analysis of an image.
    
    Extracts channels, calculates statistics for each, and returns
    a dictionary with all analysis results.
    
    Args:
        image (np.ndarray): Image to analyze
        
    Returns:
        dict: Analysis results including statistics for each channel
    """
    # Extract channels
    red, green, blue = extract_channels(image)
    
    # Calculate statistics for each channel
    red_stats = calculate_channel_statistics(red)
    green_stats = calculate_channel_statistics(green)
    blue_stats = calculate_channel_statistics(blue)
    
    # Overall image statistics
    overall_stats = calculate_channel_statistics(image)
    
    return {
        'red': red_stats,
        'green': green_stats,
        'blue': blue_stats,
        'overall': overall_stats,
        'shape': image.shape,
        'dtype': image.dtype
    }


def print_analysis(image, title="Image Analysis"):
    """
    Print formatted analysis of an image.
    
    Args:
        image (np.ndarray): Image to analyze
        title (str): Title for the analysis output
    """
    analysis = analyze_image(image)
    
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)
    
    print(f"\nImage Dimensions: {analysis['shape']}")
    print(f"Data Type: {analysis['dtype']}")
    
    # Print channel statistics
    for channel_name, stats in [('RED', analysis['red']), 
                                 ('GREEN', analysis['green']), 
                                 ('BLUE', analysis['blue'])]:
        print(f"\n{channel_name} CHANNEL STATISTICS:")
        print(f"  Min:     {stats['min']:.1f}")
        print(f"  Max:     {stats['max']:.1f}")
        print(f"  Mean:    {stats['mean']:.2f}")
        print(f"  Median:  {stats['median']:.1f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        print(f"  Variance: {stats['variance']:.2f}")


# ============================================================================
# SECTION 5: SAVING AND LOADING IMAGES
# ============================================================================

def save_image(image, filename):
    """
    Save a NumPy image array to a PNG file.
    
    TECHNICAL DETAILS:
    - PIL handles the conversion from NumPy array to image format
    - PNG is lossless (no compression artifacts)
    - Supports 8-bit channels (0-255 per channel)
    
    Args:
        image (np.ndarray): Image array to save
        filename (str): Output filename (should end with .png)
    """
    # Ensure image is uint8 format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Create PIL Image from NumPy array
    # PIL expects array in format [H, W, 3] for RGB
    pil_image = Image.fromarray(image, mode='RGB')
    
    # Save to file
    pil_image.save(filename)
    
    print(f"✓ Image saved to: {filename}")


# ============================================================================
# SECTION 6: MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of image manipulation as numerical matrices.
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "UNDERSTANDING IMAGES AS NUMERICAL MATRICES".center(78) + "║")
    print("║" + "RGB Color Channels, Pixel Manipulation, and Image Analysis".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # ========================================================================
    # PART 1: CREATE BASIC COLORED IMAGES
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 1: CREATING SOLID-COLORED IMAGES")
    print("─" * 80)
    
    print("\nCreating 100×100 images in different colors...")
    
    # Create colored images
    red_image = create_solid_color_image(100, 100, color=(255, 0, 0))
    green_image = create_solid_color_image(100, 100, color=(0, 255, 0))
    blue_image = create_solid_color_image(100, 100, color=(0, 0, 255))
    yellow_image = create_solid_color_image(100, 100, color=(255, 255, 0))
    
    print(f"Red Image shape: {red_image.shape}")
    print(f"Red Image dtype: {red_image.dtype}")
    print(f"Red Image - first pixel value: {red_image[0, 0, :]}")
    
    # Analyze red image
    print_analysis(red_image, "RED IMAGE ANALYSIS")
    
    # ========================================================================
    # PART 2: CHANNEL SEPARATION
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 2: CHANNEL EXTRACTION AND ANALYSIS")
    print("─" * 80)
    
    print("\nExtracting RGB channels from yellow image...")
    print("Note: Yellow = Red + Green, so B channel should be 0")
    
    yellow_red, yellow_green, yellow_blue = extract_channels(yellow_image)
    
    yellow_stats_red = calculate_channel_statistics(yellow_red)
    yellow_stats_green = calculate_channel_statistics(yellow_green)
    yellow_stats_blue = calculate_channel_statistics(yellow_blue)
    
    print(f"\nYellow Image Channel Statistics:")
    print(f"RED   Channel - Min: {yellow_stats_red['min']}, Max: {yellow_stats_red['max']}, Mean: {yellow_stats_red['mean']:.1f}")
    print(f"GREEN Channel - Min: {yellow_stats_green['min']}, Max: {yellow_stats_green['max']}, Mean: {yellow_stats_green['mean']:.1f}")
    print(f"BLUE  Channel - Min: {yellow_stats_blue['min']}, Max: {yellow_stats_blue['max']}, Mean: {yellow_stats_blue['mean']:.1f}")
    
    # ========================================================================
    # PART 3: IMAGE TRANSFORMATIONS
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 3: IMAGE TRANSFORMATIONS (Brighten, Darken)")
    print("─" * 80)
    
    print("\nOriginal red image analysis:")
    original_stats = analyze_image(red_image)
    print(f"Red channel mean: {original_stats['red']['mean']:.1f}")
    
    print("\nBrightening red image by 1.2×...")
    bright_red = brighten_image(red_image, factor=1.2)
    bright_stats = analyze_image(bright_red)
    print(f"Brightened red channel mean: {bright_stats['red']['mean']:.1f}")
    
    print("\nDarkening red image by 0.5×...")
    dark_red = darken_image(red_image, factor=0.5)
    dark_stats = analyze_image(dark_red)
    print(f"Darkened red channel mean: {dark_stats['red']['mean']:.1f}")
    
    # ========================================================================
    # PART 4: GRADIENT IMAGE
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 4: CREATING GRADIENT IMAGES")
    print("─" * 80)
    
    print("\nCreating a gradient image (black to white transition)...")
    gradient = create_gradient_image(100, 100)
    gradient_stats = analyze_image(gradient)
    
    print(f"Gradient image - Red channel statistics:")
    print(f"  Min: {gradient_stats['red']['min']}")
    print(f"  Max: {gradient_stats['red']['max']}")
    print(f"  Mean: {gradient_stats['red']['mean']:.2f}")
    print(f"  This represents a smooth transition from 0 (black) to 255 (white)")
    
    # ========================================================================
    # PART 5: SAVING IMAGES
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 5: SAVING IMAGES TO FILES")
    print("─" * 80)
    
    print("\nSaving images to PNG files...")
    save_image(red_image, "06_04_2026_red_image.png")
    save_image(gradient, "06_04_2026_gradient_image.png")
    save_image(bright_red, "06_04_2026_bright_red_image.png")
    save_image(dark_red, "06_04_2026_dark_red_image.png")
    
    # ========================================================================
    # PART 6: SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "─" * 80)
    print("SUMMARY: KEY CONCEPTS")
    print("─" * 80)
    
    summary = """
KEY INSIGHTS ABOUT IMAGES AS NUMERICAL MATRICES:

1. IMAGE REPRESENTATION:
   - Every image is a 3D array: [Height, Width, 3 channels]
   - Each element is an integer 0-255 (uint8 format)
   - 100×100×3 image = 30,000 numerical values

2. COLOR MODEL (RGB):
   - Red layer: controls red intensity (0-255)
   - Green layer: controls green intensity (0-255)
   - Blue layer: controls blue intensity (0-255)
   - Combining channels creates all visible colors

3. PIXEL VALUES:
   - [255,0,0] = pure red
   - [0,255,0] = pure green
   - [0,0,255] = pure blue
   - [255,255,255] = white
   - [0,0,0] = black
   - [128,128,128] = gray

4. MATRIX OPERATIONS:
   - Extract channel: img[:,:,0] gets red channel (2D matrix)
   - Brighten: multiply all values by factor > 1
   - Darken: multiply all values by factor < 1
   - Gradient: linearly interpolate values across image

5. STATISTICS ON CHANNELS:
   - Min/Max: darkest/brightest pixels
   - Mean: average pixel brightness
   - Std Dev: variability in brightness
   - All computed using NumPy vectorized operations

6. PRACTICAL APPLICATIONS:
   - Computer Vision: object detection, segmentation
   - Image Processing: filtering, enhancement, compression
   - Deep Learning: image classification, style transfer
   - Data Analysis: satellite imagery, medical imaging
    """
    
    print(summary)
    
    print("\n" + "═" * 80)
    print("DEMONSTRATION COMPLETE".center(80))
    print("═" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("File: 06_04_2026_Image_as_Numbers.py")
    print("Saved images: .png files generated with demonstrations")


if __name__ == "__main__":
    main()
