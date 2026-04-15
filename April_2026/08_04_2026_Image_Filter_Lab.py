"""
================================================================================
FILE: 08_04_2026_Image_Filter_Lab.py
TITLE: Image Processing Filters and Edge Detection
================================================================================

Comprehensive exploration of image filtering techniques including grayscale
conversion, Gaussian blur, and Canny edge detection. Demonstrates both the
mathematical principles and practical implementation of fundamental image
processing algorithms used in computer vision applications.

KEY TOPICS:
- Grayscale conversion and the luminosity formula
- Gaussian blur and kernel convolution
- Canny edge detection algorithm
- Filter visualization and comparison
- Image statistics before/after filtering

MATHEMATICAL FOUNDATIONS:
- Convolution operation: I(out) = kernel * I(in)
- Grayscale formula: Y = 0.299R + 0.587G + 0.114B
- Gaussian kernel: G(x,y) = exp(-(x²+y²)/2σ²)
- Image gradient: ∇I = [∂I/∂x, ∂I/∂y]
- Canny thresholding: non-maximum suppression, hysteresis

DEPENDENCIES:
- opencv-python (cv2): Computer vision library
- numpy: Numerical operations
- matplotlib: Visualization

INSTALLATIONS NEEDED:
pip install opencv-python numpy matplotlib

AUTHOR: Computer Vision Lab
DATE: April 8, 2026
================================================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# SECTION 1: IMAGE CREATION AND SETUP
# ============================================================================

def create_pattern_image(height=200, width=200):
    """
    Create a synthetic image with various patterns for filter testing.
    
    The image contains:
    1. Geometric shapes: circles, rectangles, lines
    2. Different colors and contrasts
    3. Edges at various orientations
    
    This pattern is useful for demonstrating filter effects because:
    - Clear edges show edge detection effectiveness
    - Various element sizes test different kernel behaviors
    - Color variations show grayscale conversion clearly
    
    Args:
        height (int): Image height
        width (int): Image width
        
    Returns:
        np.ndarray: RGB image array of shape [height, width, 3]
    """
    # Create blank white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw filled circle - demonstrates curved edges
    # cv2.circle(image, center, radius, color, thickness)
    # thickness=-1 means filled circle
    cv2.circle(image, (60, 60), 40, (255, 0, 0), -1)  # Blue circle
    
    # Draw filled rectangle - demonstrates straight edges
    # cv2.rectangle(image, top_left, bottom_right, color, thickness)
    cv2.rectangle(image, (120, 30), (180, 90), (0, 255, 0), -1)  # Green rect
    
    # Draw another circle with different properties
    cv2.circle(image, (150, 150), 30, (0, 0, 255), -1)  # Red circle
    
    # Draw rectangle outline (not filled, thickness=2)
    cv2.rectangle(image, (30, 120), (90, 180), (255, 255, 0), 2)  # Cyan outline
    
    # Draw diagonal lines - tests edge detection on various orientations
    cv2.line(image, (0, 0), (width, height), (0, 0, 0), 2)  # Black diagonal
    cv2.line(image, (width, 0), (0, height), (128, 128, 128), 2)  # Gray diagonal
    
    return image


# ============================================================================
# SECTION 2: GRAYSCALE CONVERSION
# ============================================================================

def grayscale_formula_explanation():
    """
    Documentation of the grayscale conversion formula used.
    
    MATHEMATICAL FORMULA:
    Gray = 0.299 × R + 0.587 × G + 0.114 × B
    
    WHERE:
    - R = Red channel (0-255)
    - G = Green channel (0-255)
    - B = Blue channel (0-255)
    - Gray = Output grayscale value (0-255)
    
    WEIGHT EXPLANATION:
    The weights (0.299, 0.587, 0.114) are based on human eye sensitivity:
    - Green gets highest weight (0.587): human eye most sensitive to green
    - Red gets medium weight (0.299): moderate green sensitivity
    - Blue gets lowest weight (0.114): human eye least sensitive to blue
    
    EXAMPLE CALCULATIONS:
    1. Pure red (255, 0, 0):
       Gray = 0.299*255 + 0.587*0 + 0.114*0 = 76.35 ≈ 76 (dark)
    
    2. Pure green (0, 255, 0):
       Gray = 0.299*0 + 0.587*255 + 0.114*0 = 149.69 ≈ 150 (bright)
    
    3. Pure blue (0, 0, 255):
       Gray = 0.299*0 + 0.587*0 + 0.114*255 = 29.07 ≈ 29 (very dark)
    
    4. White (255, 255, 255):
       Gray = 0.299*255 + 0.587*255 + 0.114*255 = 255 (brightest)
    
    5. Black (0, 0, 0):
       Gray = 0 (darkest)
    
    WHY NOT EQUAL WEIGHTS?
    If we used equal weights (0.333, 0.333, 0.333), green pixels would
    appear too bright and blue too dark compared to human perception.
    """
    # Return explanation as string
    return """
GRAYSCALE CONVERSION - MATHEMATICAL EXPLANATION
================================================

Formula: Gray = 0.299×R + 0.587×G + 0.114×B

Purpose: Convert 3D color image (H×W×3) to 2D grayscale (H×W)

Weights based on human luminosity perception:
- Green: 0.587 (human eye most sensitive)
- Red:   0.299 (moderate sensitivity)
- Blue:  0.114 (human eye least sensitive)

This weighted formula preserves perceived brightness better than
equal weights because it accounts for human vision characteristics.

Implementation: Each pixel processes independently:
old_pixel = [R, G, B]
new_pixel = 0.299*R + 0.587*G + 0.114*B
"""


def convert_to_grayscale(image):
    """
    Convert RGB image to grayscale using the weighted luminosity formula.
    
    ALGORITHM:
    1. Extract R, G, B channels
    2. Apply weights: Gray = 0.299R + 0.587G + 0.114B
    3. Convert to uint8 (0-255 range)
    
    COMPLEXITY:
    - Time: O(H × W) - must process every pixel
    - Space: O(H × W) - output is 2D, not 3D like input
    
    Args:
        image (np.ndarray): RGB image of shape [H, W, 3]
        
    Returns:
        np.ndarray: Grayscale image of shape [H, W]
    """
    # Extract channels
    R = image[:, :, 0].astype(np.float32)  # Red channel
    G = image[:, :, 1].astype(np.float32)  # Green channel
    B = image[:, :, 2].astype(np.float32)  # Blue channel
    
    # Apply weighted formula using NumPy broadcasting
    # This is vectorized - no Python loops needed
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Convert to uint8 (0-255)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    return gray


# ============================================================================
# SECTION 3: GAUSSIAN BLUR
# ============================================================================

def gaussian_blur_explanation():
    """
    Documentation of Gaussian blur algorithm and mathematics.
    
    MATHEMATICAL CONCEPT:
    Gaussian blur applies a Gaussian kernel to the image through convolution.
    
    CONVOLUTION OPERATION:
    For each pixel (i, j), we compute:
    Output(i,j) = sum of (Kernel(m,n) × Image(i+offset_m, j+offset_n))
    
    GAUSSIAN KERNEL MATRIX:
    The kernel is a 2D Gaussian distribution centered on the pixel:
    
    G(x,y) = (1/(2πσ²)) × exp(-(x²+y²)/(2σ²))
    
    WHERE:
    - σ (sigma) = standard deviation = blur radius
    - Larger σ = more blur (larger kernel)
    - Smaller σ = less blur (sharper kernel)
    
    EXAMPLE 5×5 GAUSSIAN KERNEL (σ=1.0):
    [0.0037  0.0147  0.0256  0.0147  0.0037]
    [0.0147  0.0586  0.0930  0.0586  0.0147]  (normalized to sum=1)
    [0.0256  0.0930  0.1458  0.0930  0.0256]
    [0.0147  0.0586  0.0930  0.0586  0.0147]
    [0.0037  0.0147  0.0256  0.0147  0.0037]
    
    HOW IT WORKS:
    1. Center kernel on each pixel
    2. Multiply kernel values by corresponding image pixels
    3. Sum all products -> output pixel
    4. Move kernel to next pixel, repeat
    
    EDGE HANDLING:
    At image borders, we use padding (zero-padding or reflect padding)
    to ensure output has same size as input.
    
    EFFECT ON IMAGE:
    - Reduces noise
    - Creates smooth transitions
    - Blurs edges (loses detail)
    - Reduces high-frequency information
    
    COMPLEXITY:
    - Time: O(H × W × K²) where K = kernel size
    - Space: O(H × W) output
    """
    return """
GAUSSIAN BLUR - MATHEMATICAL AND ALGORITHMIC EXPLANATION
===========================================================

Purpose: Smooth image using weighted averaging of neighbor pixels

Operation: 2D Convolution with Gaussian kernel

Kernel Size: Determines blur amount
- 3×3: minimal blur
- 5×5: light blur
- 9×9: moderate blur
- 25×25: heavy blur

Gaussian Distribution: Weights center pixel more heavily
- Center pixel weight: highest
- Nearby pixels: moderate weight
- Distant pixels: low weight

Effects:
- Reduces high-frequency noise
- Blurs sharp edges
- Creates smooth gradients
- Improves image smoothness
- Useful for preprocessing before edge detection
"""


def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to image using OpenCV.
    
    STANDARD LIBRARY APPROACH:
    We use cv2.GaussianBlur() from OpenCV, which is highly optimized:
    - Uses separable kernels (reduces computation)
    - Applies padding automatically
    - Handles edge cases efficiently
    
    Args:
        image (np.ndarray): Input image (can be RGB or grayscale)
        kernel_size (int): Size of Gaussian kernel (must be odd)
        sigma (float): Standard deviation of Gaussian
        
    Returns:
        np.ndarray: Blurred image (same shape as input)
    """
    # cv2.GaussianBlur(image, kernel_size, sigma)
    # kernel_size must be odd: (3,3), (5,5), (7,7), etc.
    kernel_size = int(kernel_size) if kernel_size % 2 == 1 else int(kernel_size) + 1
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return blurred


# ============================================================================
# SECTION 4: CANNY EDGE DETECTION
# ============================================================================

def canny_edge_detection_explanation():
    """
    Documentation of the Canny edge detection algorithm.
    
    CANNY ALGORITHM STEPS:
    
    Step 1: NOISE REDUCTION
    - Apply Gaussian blur to reduce noise
    - Prevents detection of noise as edges
    
    Step 2: COMPUTE GRADIENTS
    - Calculate image gradient: ∇I = [∂I/∂x, ∂I/∂y]
    - Use Sobel operators:
      * Sobel_x: detects vertical edges (left-right differences)
      * Sobel_y: detects horizontal edges (up-down differences)
    - Calculate gradient magnitude: |∇I| = sqrt(Gx² + Gy²)
    - Calculate gradient direction: θ = atan2(Gy, Gx)
    
    Step 3: NON-MAXIMUM SUPPRESSION
    - Thin edges to 1-pixel width
    - Keep only local maxima along gradient direction
    - Suppresses pixels not at edge maxima
    - Produces thin, clean edge lines
    
    Step 4: DOUBLE THRESHOLDING
    - Calculate two thresholds: T_low and T_high
    - T_low = 0.4 × max_gradient (e.g., 100)
    - T_high = 1.2 × max_gradient (e.g., 300)
    
    Pixel Classification:
    - Gradient > T_high: STRONG edge (definitely edge)
    - T_low < Gradient < T_high: WEAK edge (maybe edge)
    - Gradient < T_low: NOT edge (definitely not edge)
    
    Step 5: EDGE TRACKING BY HYSTERESIS
    - Strong edges are definitely in output
    - Weak edges included only if connected to strong edges
    - This preserves continuous edge lines
    - Creates clean, continuous edge maps
    
    OUTPUT:
    Binary image where:
    - 255 = edge pixel
    - 0 = non-edge pixel
    
    PARAMETERS:
    - threshold1 (T_low): Lower threshold
    - threshold2 (T_high): Upper threshold
    - apertureSize: Size of Sobel kernel (e.g., 3, 5, 7)
    - L2gradient: True=sqrt(Gx²+Gy²), False=|Gx|+|Gy|
    
    TUNING:
    - Higher threshold2: fewer, more confident edges
    - Lower threshold1: more edges detected, possible noise
    """
    return """
CANNY EDGE DETECTION - ALGORITHM EXPLANATION
=============================================

Five-Step Algorithm:
1. Noise Reduction (Gaussian Blur)
2. Gradient Calculation (Sobel operators)
3. Non-Maximum Suppression (thin edges)
4. Double Thresholding (classify pixels)
5. Edge Tracking by Hysteresis (connect edges)

Gradient Components:
- Gx: detects vertical edges (left-right intensity change)
- Gy: detects horizontal edges (up-down intensity change)
- Magnitude: |G| = sqrt(Gx² + Gy²)
- Direction: θ = atan2(Gy, Gx)

Double Thresholding:
- Strong edges (> T_high): definitely edges
- Weak edges (T_low to T_high): maybe edges
- Non-edges (< T_low): definitely not edges

Hysteresis:
- Weak edges kept only if connected to strong edges
- Creates clean, continuous edge lines
- Avoids fragmented edge detection

Effects:
- Detects edges at various orientations
- Produces thin, clean edge lines
- Robust to noise
- Parameter-sensitive (thresholds affect output)
"""


def apply_canny_edge_detection(image, threshold1=100, threshold2=300):
    """
    Apply Canny edge detection to image.
    
    ALGORITHM SUMMARY:
    1. Gaussian blur (reduces noise)
    2. Sobel gradients (detects intensity changes)
    3. Non-maximum suppression (thins edges)
    4. Double thresholding (classifies edges)
    5. Hysteresis (connects edge segments)
    
    Args:
        image (np.ndarray): Input image (should be grayscale or will be converted)
        threshold1 (float): Lower threshold for double thresholding
        threshold2 (float): Upper threshold for double thresholding
        
    Returns:
        np.ndarray: Binary edge map (255 = edge, 0 = non-edge)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Canny edge detection
    # cv2.Canny(image, threshold1, threshold2, apertureSize, L2gradient)
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3, L2gradient=True)
    
    return edges


# ============================================================================
# SECTION 5: IMAGE STATISTICS AND ANALYSIS
# ============================================================================

def calculate_image_statistics(image):
    """
    Calculate statistics for an image (grayscale or color).
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        dict: Dictionary with statistical measures
    """
    # Ensure image is properly formatted
    if len(image.shape) == 3:
        # For color images, calculate across all channels
        values = image.flatten()
    else:
        # For grayscale, flatten to 1D
        values = image.flatten()
    
    statistics = {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'shape': image.shape,
    }
    
    return statistics


def print_filter_statistics(original, filtered, filter_name):
    """
    Print before/after statistics for a filter.
    
    Args:
        original (np.ndarray): Original image
        filtered (np.ndarray): Filtered image
        filter_name (str): Name of the filter
    """
    orig_stats = calculate_image_statistics(original)
    filt_stats = calculate_image_statistics(filtered)
    
    print(f"\n{filter_name} FILTER STATISTICS:")
    print(f"  Original - Min: {orig_stats['min']:.1f}, Max: {orig_stats['max']:.1f}, Mean: {orig_stats['mean']:.1f}, Std: {orig_stats['std']:.1f}")
    print(f"  Filtered - Min: {filt_stats['min']:.1f}, Max: {filt_stats['max']:.1f}, Mean: {filt_stats['mean']:.1f}, Std: {filt_stats['std']:.1f}")


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def visualize_filters(original, grayscale, blurred, edges):
    """
    Create a 2×2 subplot showing original and three filtered versions.
    
    VISUALIZATION LAYOUT:
    ┌──────────────────┬──────────────────┐
    │     Original     │    Grayscale     │
    ├──────────────────┼──────────────────┤
    │     Blurred      │  Edges Detected  │
    └──────────────────┴──────────────────┘
    
    Args:
        original (np.ndarray): Original RGB image
        grayscale (np.ndarray): Grayscale version
        blurred (np.ndarray): Blurred version
        edges (np.ndarray): Edge-detected version
    """
    # Create figure with 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Plot 1: Original image
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if len(original.shape) == 3 else original, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: Grayscale
    axes[1].imshow(grayscale, cmap='gray')
    axes[1].set_title('Grayscale Conversion', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot 3: Blurred
    axes[2].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB) if len(blurred.shape) == 3 else blurred, cmap='gray')
    axes[2].set_title('Gaussian Blur (kernel=5, σ=1.0)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Plot 4: Edges
    axes[3].imshow(edges, cmap='gray')
    axes[3].set_title('Canny Edge Detection', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    # Set overall title
    plt.suptitle('Image Processing Filters Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust layout and save at high DPI
    plt.tight_layout()
    
    # Save figure at 300 DPI
    plt.savefig('08_04_2026_Image_Filter_Lab.png', dpi=300, bbox_inches='tight')
    print("\n✓ Filter comparison saved to: 08_04_2026_Image_Filter_Lab.png")
    
    # Show plot
    plt.show()


# ============================================================================
# SECTION 7: MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of image filtering techniques.
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "IMAGE PROCESSING FILTERS AND EDGE DETECTION".center(78) + "║")
    print("║" + "Grayscale, Gaussian Blur, and Canny Edge Detection".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # ========================================================================
    # PART 1: CREATE TEST IMAGE
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 1: CREATE SYNTHETIC TEST IMAGE")
    print("─" * 80)
    
    print("\nCreating 200×200 synthetic image with patterns...")
    original = create_pattern_image(200, 200)
    
    print(f"Original image shape: {original.shape}")
    print(f"Original image dtype: {original.dtype}")
    print(f"Original image value range: [{original.min()}, {original.max()}]")
    
    # ========================================================================
    # PART 2: GRAYSCALE CONVERSION
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 2: GRAYSCALE CONVERSION")
    print("─" * 80)
    
    print(grayscale_formula_explanation())
    
    print("\nConverting to grayscale using formula: Gray = 0.299R + 0.587G + 0.114B")
    grayscale = convert_to_grayscale(original)
    
    print(f"Grayscale image shape: {grayscale.shape}")
    print(f"Grayscale image dtype: {grayscale.dtype}")
    
    print_filter_statistics(original, grayscale, "GRAYSCALE CONVERSION")
    
    # ========================================================================
    # PART 3: GAUSSIAN BLUR
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 3: GAUSSIAN BLUR")
    print("─" * 80)
    
    print(gaussian_blur_explanation())
    
    print("\nApplying Gaussian blur (kernel size=5, σ=1.0)...")
    blurred = apply_gaussian_blur(original, kernel_size=5, sigma=1.0)
    
    print(f"Blurred image shape: {blurred.shape}")
    print(f"Blurred image dtype: {blurred.dtype}")
    
    print_filter_statistics(original, blurred, "GAUSSIAN BLUR")
    
    # ========================================================================
    # PART 4: CANNY EDGE DETECTION
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 4: CANNY EDGE DETECTION")
    print("─" * 80)
    
    print(canny_edge_detection_explanation())
    
    print("\nApplying Canny edge detection (T_low=100, T_high=300)...")
    edges = apply_canny_edge_detection(original, threshold1=100, threshold2=300)
    
    print(f"Edge map shape: {edges.shape}")
    print(f"Edge map dtype: {edges.dtype}")
    print(f"Edge map value range: [{edges.min()}, {edges.max()}]")
    print(f"Percentage of edge pixels: {(edges > 0).sum() / edges.size * 100:.2f}%")
    
    print_filter_statistics(grayscale, edges, "CANNY EDGE DETECTION")
    
    # ========================================================================
    # PART 5: VISUALIZATION
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 5: VISUALIZATION AND EXPORT")
    print("─" * 80)
    
    print("\nCreating 2×2 subplot comparison...")
    visualize_filters(original, grayscale, blurred, edges)
    
    # ========================================================================
    # PART 6: SUMMARY AND INSIGHTS
    # ========================================================================
    print("\n" + "─" * 80)
    print("SUMMARY: KEY INSIGHTS")
    print("─" * 80)
    
    summary = """
FILTER CHARACTERISTICS AND APPLICATIONS:

1. GRAYSCALE CONVERSION
   Purpose: Convert 3D RGB to 2D brightness for processing
   Formula: Gray = 0.299R + 0.587G + 0.114B
   Advantages: Reduces computation, preserves luminosity
   Applications: Preprocessing, edge detection, analysis

2. GAUSSIAN BLUR
   Purpose: Smooth image, reduce noise
   Method: Weighted averaging with Gaussian kernel
   Advantages: Preserves edges better than mean filtering
   Applications: Noise reduction, preprocessing, smoothing
   Parameter tuning:
   - Larger kernel → stronger blur
   - Higher sigma → slower transition

3. CANNY EDGE DETECTION
   Purpose: Identify image edges and boundaries
   Algorithm: 5-step process with non-max suppression
   Advantages: Thin edges, low false positives/negatives
   Applications: Object detection, boundary finding, segmentation
   Parameter tuning:
   - Higher T_low → fewer edges, less noise
   - Lower T_high → more edges, possible noise

FILTER CHAIN FOR TYPICAL EDGE DETECTION PIPELINE:
Original Image → Grayscale → Gaussian Blur → Canny → Edge Map

This sequence:
1. Reduces color information (grayscale)
2. Removes noise (blur)
3. Detects edges (canny)
4. Produces clean edge map

REAL-WORLD APPLICATIONS:
- Object detection: Find objects by their boundaries
- Lane detection: Driving assistance systems
- Medical imaging: Identify abnormalities
- Quality control: Manufacturing inspection
- Document scanning: Text boundary detection
    """
    
    print(summary)
    
    print("\n" + "═" * 80)
    print("FILTER LAB DEMONSTRATION COMPLETE".center(80))
    print("═" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Output file: 08_04_2026_Image_Filter_Lab.png (300 DPI)")


if __name__ == "__main__":
    # Run the demonstration
    main()
