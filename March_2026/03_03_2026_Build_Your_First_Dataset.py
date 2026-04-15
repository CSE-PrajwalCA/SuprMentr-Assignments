"""
================================================================================
03_03_2026_Build_Your_First_Dataset.py - Build Your First Dataset
================================================================================

TITLE: Build Your First Dataset and Explore Linear Regression

ASSIGNMENT OVERVIEW:
This comprehensive assignment guides students through the fundamental process of 
creating synthetic datasets and performing exploratory data analysis using linear 
regression. Students will learn how to generate correlated numerical data, visualize 
relationships between variables, and build their first machine learning model using 
scikit-learn. The project emphasizes understanding the mathematical foundations of 
linear regression while building practical skills in data generation, exploration, 
and visualization using industry-standard Python libraries. By the end of this 
assignment, students will have a complete end-to-end pipeline from data generation 
to model interpretation and visualization export.

LEARNING OBJECTIVES:
1. Understand how to generate synthetic datasets using NumPy with reproducible 
   randomization through seed management for research reproducibility
2. Create correlated variables to simulate real-world relationships between features 
   and understand how to add realistic noise to synthetic data
3. Perform data validation and quality checks on generated datasets including null 
   checks, range validation, and descriptive statistics
4. Calculate Pearson correlation coefficient to quantify the strength and direction 
   of linear relationships between variables
5. Create professional scatter plots with regression lines using Matplotlib and 
   understand plot customization options
6. Implement linear regression models using scikit-learn's LinearRegression class 
   and understand the sklearn pipeline
7. Interpret model performance metrics including R² score and regression coefficients 
   in business context
8. Derive and interpret the equation of the fitted regression line (y = mx + b) and 
   make predictions using the model
9. Understand the mathematical foundations of ordinary least squares (OLS) regression 
   and the concept of residuals
10. Practice generating publication-quality visualizations and exporting them as 
    high-resolution PNG files for reports and presentations

MATHEMATICAL CONCEPTS COVERED:
- Linear Regression: Finding the best-fit line through scattered data points using 
  the least squares method
- Pearson Correlation Coefficient: Measuring the strength of linear relationships 
  between variables on a scale from -1 to 1
- R² Score (Coefficient of Determination): Evaluating what proportion of variance 
  in the dependent variable is explained by the model
- Ordinary Least Squares (OLS): The mathematical method behind regression fitting 
  that minimizes sum of squared residuals
- Regression Equation: The formula y = mx + b where m is slope (rate of change) 
  and b is intercept (y-value at x=0)

KEY LIBRARIES USED:
- NumPy: For numerical computations and synthetic data generation with random distributions
- Matplotlib: For creating, customizing, and saving publication-quality visualizations
- Scikit-learn: For machine learning model implementation and performance evaluation
- SciPy: For statistical calculations including Pearson correlation and p-values

================================================================================
"""

# Import necessary libraries for data generation, visualization, and ML
import numpy as np  # For numerical operations and data generation
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.linear_model import LinearRegression  # For implementing linear regression model
from scipy.stats import pearsonr  # For calculating Pearson correlation coefficient
import warnings  # For suppressing warnings if needed

# ============================================================================
# SECTION 1: DATA GENERATION WITH SEED FOR REPRODUCIBILITY
# ============================================================================

# Set the random seed to 42 for reproducible results across different runs
np.random.seed(42)  # Using seed 42 ensures same results every time program runs

# Define the number of samples in our dataset
num_samples = 100  # Creating a dataset with 100 student records

# Generate study hours: uniformly distributed between 1 and 10 hours
study_hours = np.random.uniform(1, 10, num_samples)  # Students study between 1-10 hours

# Generate marks with positive correlation to study hours
# Base marks (intercept) around 40, with additional points based on study hours
base_marks = 40  # Baseline mark without studying
marks_coefficient = 8  # Each hour of study adds approximately 8 marks
noise = np.random.normal(0, 5, num_samples)  # Add realistic variation (mean=0, std=5)
marks = base_marks + (marks_coefficient * study_hours) + noise  # Calculate marks with noise

# Ensure marks are within realistic bounds [0, 100]
marks = np.clip(marks, 0, 100)  # Clip marks to be between 0 and 100

# Create a 2D array combining study_hours and marks for easier manipulation
dataset = np.column_stack((study_hours, marks))  # Stack hours and marks as columns

print("\n" + "="*80)
print("DATASET GENERATION COMPLETE")
print("="*80)

# ============================================================================
# SECTION 2: DATA VALIDATION AND SUMMARY STATISTICS
# ============================================================================

# Verify dataset dimensions and check for any missing or invalid values
print("\n" + "-"*80)
print("DATASET VALIDATION AND SUMMARY STATISTICS")
print("-"*80)

# Display dataset shape information
print(f"\nDataset Shape: {dataset.shape[0]} samples × {dataset.shape[1]} features")  # Print dimensions
print(f"Expected shape: (100, 2)")  # Expected dimensions

# Check for missing values (NaN) in the dataset
missing_values = np.isnan(dataset).sum()  # Count NaN values
print(f"Missing Values (NaN): {missing_values}")  # Should be 0

# Check for infinite values in the dataset
infinite_values = np.isinf(dataset).sum()  # Count infinite values
print(f"Infinite Values: {infinite_values}")  # Should be 0

# Validate that all values are numeric and within expected ranges
study_hours_valid = np.all((study_hours >= 1) & (study_hours <= 10))  # Check range
marks_valid = np.all((marks >= 0) & (marks <= 100))  # Check marks are in [0,100]
print(f"\nData Validation Results:")
print(f"  Study Hours in valid range [1, 10]: {study_hours_valid}")  # Validation status
print(f"  Marks in valid range [0, 100]: {marks_valid}")  # Validation status

# Calculate and display summary statistics for study_hours
print(f"\n--- STUDY HOURS STATISTICS ---")
print(f"  Mean (Average): {np.mean(study_hours):.2f} hours")  # Average study hours
print(f"  Median (Middle): {np.median(study_hours):.2f} hours")  # Median study hours
print(f"  Std Dev (Spread): {np.std(study_hours):.2f} hours")  # Standard deviation
print(f"  Min (Minimum): {np.min(study_hours):.2f} hours")  # Minimum value
print(f"  Max (Maximum): {np.max(study_hours):.2f} hours")  # Maximum value

# Calculate and display summary statistics for marks
print(f"\n--- MARKS STATISTICS ---")
print(f"  Mean (Average): {np.mean(marks):.2f} marks")  # Average marks
print(f"  Median (Middle): {np.median(marks):.2f} marks")  # Median marks
print(f"  Std Dev (Spread): {np.std(marks):.2f} marks")  # Standard deviation
print(f"  Min (Minimum): {np.min(marks):.2f} marks")  # Minimum value
print(f"  Max (Maximum): {np.max(marks):.2f} marks")  # Maximum value

# Display first and last few samples from the dataset
print(f"\n--- FIRST 5 SAMPLES ---")
for i in range(5):  # Iterate through first 5 records
    print(f"  Sample {i+1}: Study Hours = {study_hours[i]:.2f}, Marks = {marks[i]:.2f}")  # Print record

print(f"\n--- LAST 5 SAMPLES ---")
for i in range(95, 100):  # Iterate through last 5 records
    print(f"  Sample {i+1}: Study Hours = {study_hours[i]:.2f}, Marks = {marks[i]:.2f}")  # Print record

# ============================================================================
# SECTION 3: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Calculate Pearson correlation coefficient between study_hours and marks
correlation_coefficient, p_value = pearsonr(study_hours, marks)  # Calculate correlation and p-value

print(f"\n--- PEARSON CORRELATION RESULTS ---")
print(f"  Correlation Coefficient: {correlation_coefficient:.4f}")  # Display correlation value
print(f"  P-Value (Statistical Significance): {p_value:.2e}")  # Display p-value

print(f"\nInterpretation:")
print(f"  - Correlation close to 1.0 indicates strong positive relationship")  # Positive correlation range
print(f"  - Correlation close to 0.0 indicates weak or no relationship")  # No correlation range
print(f"  - P-value < 0.05 typically indicates statistically significant correlation")  # Significance threshold

# Provide qualitative assessment of correlation strength
if abs(correlation_coefficient) >= 0.7:  # Strong correlation threshold
    strength = "STRONG"
elif abs(correlation_coefficient) >= 0.5:  # Moderate correlation threshold
    strength = "MODERATE"
elif abs(correlation_coefficient) >= 0.3:  # Weak correlation threshold
    strength = "WEAK"
else:  # Very weak or no correlation
    strength = "VERY WEAK"

direction = "POSITIVE" if correlation_coefficient > 0 else "NEGATIVE"  # Determine direction
print(f"  - Relationship Strength: {strength} {direction}")  # Print assessment

# ============================================================================
# SECTION 4: LINEAR REGRESSION MODEL
# ============================================================================

print("\n" + "="*80)
print("LINEAR REGRESSION MODEL")
print("="*80)

# Reshape study_hours to be a 2D array (required by scikit-learn)
X = study_hours.reshape(-1, 1)  # Reshape from (100,) to (100, 1) for sklearn

# Create a LinearRegression model object
model = LinearRegression()  # Initialize the regression model

# Fit the model to our data (find the best-fit line)
model.fit(X, marks)  # Train the model using X (study_hours) and marks (target)

# Extract the model parameters
slope = model.coef_[0]  # Extract slope (m) from trained model
intercept = model.intercept_  # Extract intercept (b) from trained model

# Calculate predictions using the fitted model
predictions = model.predict(X)  # Get predicted marks for all study hours

# Calculate R² score (coefficient of determination)
r2_score = model.score(X, marks)  # Calculate R² score (0 to 1, higher is better)

print(f"\n--- REGRESSION EQUATION ---")
print(f"  Linear Regression Equation: y = {slope:.4f}x + {intercept:.4f}")  # Print equation
print(f"  Interpretation:")
print(f"    - For every additional hour of study, marks increase by {slope:.4f}")  # Slope interpretation
print(f"    - Without any study (x=0), expected marks would be {intercept:.4f}")  # Intercept interpretation

print(f"\n--- MODEL PERFORMANCE METRICS ---")
print(f"  R² Score: {r2_score:.4f}")  # Print R² value
print(f"  Interpretation: {r2_score*100:.2f}% of variance in marks is explained by study hours")  # R² explanation

# ============================================================================
# SECTION 5: MATHEMATICAL EXPLANATION OF LINEAR REGRESSION
# ============================================================================

print("\n" + "="*80)
print("MATHEMATICAL FOUNDATION OF LINEAR REGRESSION")
print("="*80)

print(f"""
LINEAR REGRESSION BASICS:
Linear regression finds the best-fit line through data points by minimizing the sum
of squared errors (distances between actual and predicted values). This method is
called Ordinary Least Squares (OLS) and is the foundation of most statistical models.

THE ORDINARY LEAST SQUARES (OLS) METHOD:
The model aims to minimize: Σ(actual_y - predicted_y)²
This means we want to find the line where the sum of all squared vertical distances
(called residuals) from points to the line is as small as possible.

KEY EQUATION:
  y = mx + b
  where:
    m = slope (change in y per unit change in x)
    b = intercept (value of y when x = 0)
    x = independent variable (study hours)
    y = dependent variable (marks)

SLOPE CALCULATION (simplified):
  m = Correlation × (StdDev(y) / StdDev(x))
  The slope is proportional to the correlation coefficient and standard deviations.

INTERCEPT CALCULATION:
  b = Mean(y) - m × Mean(x)
  The intercept represents the expected value of y when x equals zero.

R² SCORE EXPLANATION:
  R² represents the proportion of variance in the dependent variable that is
  predictable from the independent variable(s). 
  - R² = 1: Perfect fit (all points lie exactly on the line)
  - R² = 0: Model explains no variance (no better than using mean)
  - R² ranges from 0 to 1 (for this type of analysis)
  - R² of 0.85 means 85% of variance in marks is explained by study hours

IN THIS DATASET:
  - Slope ({slope:.4f}): Each additional study hour increases expected marks by {slope:.4f}
  - Intercept ({intercept:.4f}): Baseline mark without studying
  - R² ({r2_score:.4f}): The model explains {r2_score*100:.2f}% of mark variations
  - This indicates a strong relationship between study hours and marks
""")

# ============================================================================
# SECTION 6: VISUALIZATION AND PLOTTING
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATION")
print("="*80)

# Create a new figure with specific size for better visibility
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches for clarity

# Scatter plot of actual data points
plt.scatter(study_hours, marks, alpha=0.6, s=80, color='blue', label='Actual Data Points')  # Plot actual data

# Plot the regression line
study_hours_line = np.linspace(study_hours.min(), study_hours.max(), 100)  # Generate line points
marks_line = slope * study_hours_line + intercept  # Calculate marks for line points
plt.plot(study_hours_line, marks_line, color='red', linewidth=2.5, label='Regression Line')  # Plot line

# Add labels and title
plt.xlabel('Study Hours', fontsize=12, fontweight='bold')  # X-axis label with formatting
plt.ylabel('Marks Obtained', fontsize=12, fontweight='bold')  # Y-axis label with formatting
plt.title('Linear Regression: Study Hours vs Marks\nBuild Your First Dataset', 
          fontsize=14, fontweight='bold')  # Main title with formatting

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')  # Add grid with dashed lines and transparency

# Create text box with model details
textstr = f'Equation: y = {slope:.4f}x + {intercept:.4f}\nR² Score: {r2_score:.4f}\nCorrelation: {correlation_coefficient:.4f}'  # Create text string
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)  # Define box styling properties
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)  # Add text box to plot

# Add legend to the plot
plt.legend(loc='lower right', fontsize=10)  # Position legend at lower right corner

# Tight layout to prevent label cutoff
plt.tight_layout()  # Adjust spacing automatically to prevent overlapping

# Save the figure as PNG file
output_filename = '03_03_2026_Build_Your_First_Dataset.png'  # Name of output image file
plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # Save at high resolution (300 DPI)
print(f"\nVisualization saved as: {output_filename}")  # Confirm file save

# Display the plot
plt.show()  # Display the plot in the window

# ============================================================================
# SECTION 7: FINAL SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY AND CONCLUSIONS")
print("="*80)

print(f"""
KEY FINDINGS:
1. Dataset: Successfully created 100 synthetic samples with correlated variables
2. Correlation: Strong positive correlation of {correlation_coefficient:.4f} between
   study hours and marks, indicating clear relationship
3. Model: Linear regression successfully fits the data with good accuracy
4. Equation: y = {slope:.4f}x + {intercept:.4f}
5. Performance: R² = {r2_score:.4f} ({r2_score*100:.2f}% variance explained)

PRACTICAL INTERPRETATION:
- Students who study more hours consistently tend to score higher marks
- The relationship is approximately linear within the study range [1-10 hours]
- For every additional hour of study, a student can expect to gain {slope:.4f} marks
- The model's R² value of {r2_score:.4f} indicates a {'strong' if r2_score > 0.7 else 'moderate' if r2_score > 0.5 else 'weak'} fit
- Some variation in marks is due to other factors not captured in study hours alone
- A small amount of variation is natural and expected in real-world data

LEARNING OUTCOMES ACHIEVED:
✓ Created and validated a synthetic dataset with 100 samples
✓ Performed exploratory data analysis with multiple statistics
✓ Calculated correlation statistics and tested significance
✓ Built a linear regression model using scikit-learn
✓ Interpreted model coefficients in practical context
✓ Evaluated model performance using R² metric
✓ Created professional visualizations with annotations
✓ Understood the mathematics behind regression methods
✓ Exported high-quality PNG file for presentations
✓ Practiced complete data science pipeline from start to finish
""")

print("="*80)
print("PROGRAM EXECUTION COMPLETED SUCCESSFULLY")
print("="*80 + "\n")
