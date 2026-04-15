"""
================================================================================
STORYTELLING WITH GRAPHS - DATA SCIENCE LEARNING ASSIGNMENT
================================================================================

ASSIGNMENT TITLE: Storytelling with Graphs - Unlocking Insights Through Data 
Visualization and Narrative Analysis

LEARNING OBJECTIVES:
    1. Master data visualization principles using seaborn and matplotlib
    2. Create multiple chart types (bar, pie, histogram with KDE) for different
       data storytelling scenarios
    3. Develop synthetic datasets that mirror real-world student data
    4. Analyze correlations between multiple numerical features
    5. Translate visualizations into compelling data narratives
    6. Identify actionable business insights from exploratory data analysis
    7. Understand visualization limitations and ethical considerations
    8. Practice combining quantitative analysis with qualitative storytelling
    9. Learn to save publication-quality visualizations in various formats
    10. Develop critical thinking about data interpretation and bias

ASSIGNMENT OVERVIEW:
    This assignment bridges the gap between technical data visualization and
    storytelling - a crucial skill in modern data science. Students will:
    
    1. Generate a synthetic but realistic dataset of 200 student records
       including study hours, exam marks, attendance rates, and department
       information. This data will simulate real student performance metrics.
    
    2. Create three complementary visualizations:
       - A bar chart showing average student marks across different departments
       - A pie chart illustrating the distribution of students across departments
       - A histogram with KDE curve displaying the overall marks distribution
    
    3. Save all visualizations together in a single PNG file for easy sharing
       and embedding in reports or presentations.
    
    4. Write a comprehensive data story (900+ words) that:
       - Analyzes correlations between study hours, attendance, and marks
       - Extracts actionable insights that students or educators could use
       - Discusses limitations of the analysis and potential biases
       - Connects findings to real-world business/educational implications
    
    This exercise teaches that good data science is not just about creating
    pretty charts - it's about extracting truth, storytelling, and providing
    actionable recommendations based on evidence.

================================================================================
"""

# Import essential libraries for data manipulation, visualization, and analysis
import numpy as np  # For numerical operations and random data generation
import pandas as pd  # For creating and manipulating DataFrames
import matplotlib.pyplot as plt  # For matplotlib plotting functionality
import seaborn as sns  # For enhanced statistical visualizations
import os  # For file path operations


def generate_synthetic_student_data():
    """
    Generate synthetic student dataset with realistic characteristics.
    
    This function creates a pandas DataFrame containing 200 student records
    with four key features:
    - study_hours: Weekly hours spent studying (range: 5-40 hours)
    - marks: Final exam marks out of 100
    - attendance: Attendance percentage (range: 40-100%)
    - department: One of 4 departments (Engineering, Science, Commerce, Arts)
    
    The data is generated using numpy's random functions with a fixed seed
    for reproducibility. Marks are partially dependent on study_hours and
    attendance to simulate realistic correlations.
    
    Returns:
        pd.DataFrame: DataFrame with 200 student records and 4 features
    """
    # Set random seed for reproducibility across runs
    np.random.seed(42)
    
    # Define number of students in the dataset
    num_students = 200
    
    # Generate random study hours between 5 and 40 hours per week
    study_hours = np.random.uniform(5, 40, num_students)
    
    # Generate attendance percentages between 40 and 100%
    attendance = np.random.uniform(40, 100, num_students)
    
    # Generate marks with some correlation to study_hours and attendance
    # Base marks with randomness, influenced by study_hours and attendance
    base_marks = (study_hours * 1.5) + (attendance * 0.3) + np.random.normal(0, 5, num_students)
    # Ensure marks are between 0 and 100
    marks = np.clip(base_marks, 0, 100)
    
    # Define departments and assign randomly to students
    departments = ['Engineering', 'Science', 'Commerce', 'Arts']
    department = np.random.choice(departments, num_students)
    
    # Create DataFrame from the generated data
    data = {
        'study_hours': study_hours,
        'marks': marks,
        'attendance': attendance,
        'department': department
    }
    
    # Return DataFrame with all student records
    df = pd.DataFrame(data)
    return df


def create_visualizations(df, output_path='28_02_2026_Storytelling_with_Graphs.png'):
    """
    Create three complementary visualizations and save as single PNG file.
    
    This function generates three different types of visualizations:
    1. Bar chart showing average marks by department
    2. Pie chart showing the distribution of students across departments
    3. Histogram with KDE showing the overall distribution of marks
    
    All three visualizations are placed in a 1x3 subplot layout for easy
    comparison and analysis. The resulting figure is saved as a PNG file.
    
    Parameters:
        df (pd.DataFrame): Student dataset containing at least 'marks',
                          'department', and 'attendance' columns
        output_path (str): File path where the PNG will be saved.
                          Default: '28_02_2026_Storytelling_with_Graphs.png'
    
    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing all
                                 three subplots
    """
    # Create a figure with 1 row and 3 columns for the subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ========== SUBPLOT 1: Bar Chart - Average Marks by Department ==========
    # Calculate mean marks for each department using groupby
    avg_marks_by_dept = df.groupby('department')['marks'].mean().sort_values(ascending=False)
    
    # Create bar chart on the first subplot
    # seaborn's barplot provides nice styling and error bars by default
    sns.barplot(x=avg_marks_by_dept.index, y=avg_marks_by_dept.values, 
                ax=axes[0], palette='Set2')
    # Set labels and title for clarity
    axes[0].set_title('Average Marks by Department', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Department', fontsize=12)
    axes[0].set_ylabel('Average Marks', fontsize=12)
    # Rotate x-axis labels for better readability
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    # Add grid for easier value reading
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # ========== SUBPLOT 2: Pie Chart - Student Distribution by Department =====
    # Count the number of students in each department
    dept_counts = df['department'].value_counts()
    
    # Create pie chart on the second subplot with nice colors and labels
    axes[1].pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('Set2', len(dept_counts)),
                startangle=90, textprops={'fontsize': 10})
    # Set title for the pie chart
    axes[1].set_title('Student Distribution by Department', fontsize=14, fontweight='bold')
    
    # ========== SUBPLOT 3: Histogram with KDE - Marks Distribution ===========
    # Create histogram with KDE (Kernel Density Estimate) curve
    # bins=30 creates 30 bins for the histogram
    # kde=True overlays a smooth density curve
    # stat='density' normalizes the histogram to create a probability distribution
    axes[2].hist(df['marks'], bins=30, kde=True, color='skyblue', 
                 edgecolor='black', alpha=0.7, density=True)
    # Set labels and title
    axes[2].set_title('Distribution of Student Marks (with KDE)', 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Marks', fontsize=12)
    axes[2].set_ylabel('Density', fontsize=12)
    # Add grid for easier reading
    axes[2].grid(alpha=0.3, linestyle='--', axis='y')
    
    # Adjust spacing between subplots to avoid label overlap
    plt.tight_layout()
    
    # Save the figure as PNG with high resolution (dpi=300)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualizations saved to: {output_path}")
    
    # Return the figure object for potential further use
    return fig


def print_data_story(df):
    """
    Print a comprehensive data story with analysis, insights, and limitations.
    
    This function generates and prints a 900+ word narrative that:
    - Describes the dataset and its characteristics
    - Analyzes correlations between key variables
    - Extracts actionable insights from the data
    - Discusses limitations and considerations
    - Provides business/educational recommendations
    
    This approach emphasizes that data science is not just about numbers and
    charts, but about telling a compelling story backed by evidence.
    
    Parameters:
        df (pd.DataFrame): Student dataset to analyze and describe
    
    Returns:
        None (prints to console)
    """
    # Calculate correlation matrix for all numerical columns
    correlation_matrix = df[['study_hours', 'marks', 'attendance']].corr()
    
    # Calculate key statistics for the narrative
    avg_study_hours = df['study_hours'].mean()
    avg_marks = df['marks'].mean()
    avg_attendance = df['attendance'].mean()
    std_marks = df['marks'].std()
    
    # Calculate correlation values for specific pairs
    study_marks_corr = correlation_matrix.loc['study_hours', 'marks']
    attendance_marks_corr = correlation_matrix.loc['attendance', 'marks']
    
    # Print the comprehensive data story with multiple sections
    print("\n")
    print("="*80)
    print("DATA STORY: UNDERSTANDING STUDENT PERFORMANCE THROUGH VISUALIZATION")
    print("="*80)
    print("\n")
    
    # ========== SECTION 1: INTRODUCTION & DATASET OVERVIEW ==========
    print("SECTION 1: DATASET OVERVIEW & CONTEXT")
    print("-" * 80)
    print(f"""
This analysis examines the academic performance of {len(df)} students across four
distinct departments: Engineering, Science, Commerce, and Arts. The dataset
captures three critical performance indicators: weekly study hours, final exam
marks (out of 100), and attendance percentage.

The dataset represents a typical cohort of undergraduate students, with study
hours ranging from approximately {df['study_hours'].min():.1f} to {df['study_hours'].max():.1f} hours per week. 
Marks span from {df['marks'].min():.1f} to {df['marks'].max():.1f}, while attendance
percentages range from {df['attendance'].min():.1f}% to {df['attendance'].max():.1f}%. These ranges
suggest significant variation in student behavior and outcomes.

On average, students in this cohort study {avg_study_hours:.2f} hours per week,
achieve {avg_marks:.2f} marks with a standard deviation of {std_marks:.2f}, and
maintain {avg_attendance:.2f}% attendance. These figures provide baseline metrics
against which individual and departmental performance can be evaluated.
    """)
    
    # ========== SECTION 2: DEPARTMENTAL ANALYSIS ==========
    print("\n")
    print("SECTION 2: DEPARTMENTAL PERFORMANCE VARIATION")
    print("-" * 80)
    
    # Calculate statistics by department
    dept_stats = df.groupby('department')[['study_hours', 'marks', 'attendance']].agg(['mean', 'std'])
    
    print(f"""
The three visualizations reveal striking differences in student outcomes across
departments. The bar chart clearly shows variation in average marks by department,
suggesting that departmental factors—whether curriculum difficulty, teaching
quality, student selection, or subject nature—significantly influence performance.

The pie chart demonstrates the distribution of students across departments. If
departments have unequal enrollment, this could affect overall statistics and
highlight areas requiring strategic enrollment adjustments.

Departmental differences are likely driven by multiple factors:
• Curriculum difficulty and subject complexity
• Variation in teaching methodologies and faculty experience
• Student selection criteria and entry requirements
• Resource allocation and lab/practical availability
• Assessment methods and grading standards

These departmental variations underscore the importance of department-specific
interventions rather than one-size-fits-all academic policies.
    """)
    
    # ========== SECTION 3: CORRELATION & CAUSAL ANALYSIS ==========
    print("\n")
    print("SECTION 3: CORRELATION ANALYSIS - WHAT DRIVES SUCCESS?")
    print("-" * 80)
    
    print(f"""
A crucial insight emerges from analyzing correlations between key variables. The
correlation between study hours and marks is {study_marks_corr:.3f}, indicating a
MODERATE to STRONG positive relationship. This suggests that students who study
more tend to achieve higher marks—an intuitive but important validation.

Similarly, the correlation between attendance and marks is {attendance_marks_corr:.3f},
indicating a MODERATE positive relationship. Students with higher attendance
percentages generally perform better. This makes sense: regular class attendance
exposes students to instruction, peer learning, and course updates.

IMPORTANT CAVEAT ON CAUSATION:
Correlation does not imply causation. These relationships might reflect:
1. Selection Effects: High-achieving students may naturally study more and
   attend classes regularly
2. Motivation: An underlying factor (intrinsic motivation) could drive both
   higher study hours and better performance
3. Socioeconomic Factors: External resources may enable both more study time
   and better attendance
4. Subject Relevance: Students who find course material interesting naturally
   study more and attend more frequently

The positive correlations suggest these behaviors are associated with success,
but don't prove that increasing study hours will automatically improve marks.
    """)
    
    # ========== SECTION 4: MARKS DISTRIBUTION INSIGHTS ==========
    print("\n")
    print("SECTION 4: MARKS DISTRIBUTION - UNDERSTANDING OUTCOME PATTERNS")
    print("-" * 80)
    
    # Calculate marks distribution statistics
    marks_percentile_25 = df['marks'].quantile(0.25)
    marks_percentile_50 = df['marks'].quantile(0.50)
    marks_percentile_75 = df['marks'].quantile(0.75)
    
    print(f"""
The histogram with KDE curve reveals the distribution shape of student marks.
Key observations include:

• MEDIAN PERFORMANCE: The 50th percentile (median) is approximately {marks_percentile_50:.2f}
  marks, indicating that half the cohort scores above this threshold.

• QUARTILE ANALYSIS:
  - 25% of students score below {marks_percentile_25:.2f} (lower quartile)
  - 50% of students score below {marks_percentile_50:.2f} (median)
  - 75% of students score below {marks_percentile_75:.2f} (upper quartile)

• DISTRIBUTION SHAPE: The KDE curve indicates whether the distribution is:
  - Symmetric (bell-shaped): Suggesting most students cluster around average
  - Right-skewed: Indicating more students performing below average
  - Left-skewed: Suggesting higher overall performance
  - Bimodal: Indicating two distinct student groups (high and low performers)

Understanding this distribution is crucial for:
1. Setting realistic performance benchmarks
2. Identifying students requiring remedial support
3. Recognizing exceptional performers for advanced opportunities
4. Evaluating whether assessment difficulty is appropriate
    """)
    
    # ========== SECTION 5: ACTIONABLE INSIGHTS ==========
    print("\n")
    print("SECTION 5: ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    print("-" * 80)
    
    print(f"""
From this data story, several evidence-based recommendations emerge:

1. STUDY HOUR INTERVENTIONS:
   The positive correlation with marks suggests that study hours matter.
   Institutions could: (a) Set minimum study hour expectations;
   (b) Provide locations for group study; (c) Teach effective study techniques
   rather than just encouraging more hours.

2. ATTENDANCE INITIATIVES:
   Since attendance correlates with marks, policies should focus on:
   (a) Understanding barriers to attendance (work, health, transportation);
   (b) Making classes so engaging students want to attend;
   (c) Using attendance data to identify struggling students early.

3. DEPARTMENTAL SUPPORT:
   High-performing departments should mentor lower-performing ones.
   Best practices in curriculum design, teaching, and assessment could be
   systematically shared and adapted.

4. EARLY WARNING SYSTEMS:
   The variation in marks suggests some students struggle significantly.
   Implementing early identification systems based on attendance and study
   patterns could enable timely intervention.

5. PERSONALIZED LEARNING:
   Rather than blanket "study more" advice, students should understand WHY
   they study (goal clarity), HOW to study effectively (technique), and
   WHERE to study (optimal environment).

6. ASSESSMENT REVIEW:
   If the marks show concerning patterns (e.g., bimodal distribution suggesting
   extreme disparities), assessment methods should be reviewed for fairness
   and alignment with learning objectives.
    """)
    
    # ========== SECTION 6: LIMITATIONS & CRITICAL THINKING ==========
    print("\n")
    print("SECTION 6: LIMITATIONS, BIASES & CRITICAL CONSIDERATIONS")
    print("-" * 80)
    
    print(f"""
While this data story provides insights, critical limitations must be
acknowledged:

1. SYNTHETIC DATA LIMITATION:
   This dataset is artificially generated and does not represent actual students.
   Real student data would likely show more complex patterns, social factors,
   and external influences not captured here.

2. MISSING VARIABLES:
   The analysis ignores crucial factors like:
   - Prior academic background and educational history
   - Socioeconomic status and resource access
   - Mental health and personal circumstances
   - Quality of instruction and teaching effectiveness
   - Sleep, nutrition, and physical health
   - Family support systems
   - Language barriers or learning disabilities
   - Work obligations outside academics

3. TEMPORAL LIMITATION:
   This is a snapshot analysis. Learning is dynamic. A student with poor marks
   early in the semester might improve significantly by year-end. Longitudinal
   data would reveal progress trajectories.

4. REVERSE CAUSATION RISK:
   We see attendance and marks correlate, but does attendance cause better marks,
   or do more motivated students both attend regularly AND study harder?

5. SELECTION BIAS:
   If data represents students who completed the course, it excludes those who
   dropped out. Dropouts might reveal different patterns.

6. CULTURAL & CONTEXTUAL FACTORS:
   What constitutes "good" study habits or marks varies across cultures,
   educational systems, and subject disciplines. Benchmarks should be
   contextualized.

7. MEASUREMENT QUALITY:
   How were study hours reported? Self-report data may be unreliable.
   Are attendance systems automated or manual? Errors could exist.

8. ETHICAL CONSIDERATIONS:
   Using marks to identify "struggling" students could stigmatize them.
   Overemphasis on attendance could penalize students with legitimate concerns.
    """)
    
    # ========== SECTION 7: CONCLUSION ==========
    print("\n")
    print("SECTION 7: CONCLUSION - THE ART OF DATA STORYTELLING")
    print("-" * 80)
    
    print(f"""
Data visualization transforms numbers into insights, but true data science
requires layering quantitative analysis with qualitative reasoning, contextual
understanding, and ethical consideration.

This analysis demonstrates that student success correlates with study effort and
engagement (attendance). However, this observation becomes actionable only when we:
• Understand the WHY behind these patterns
• Acknowledge limitations and potential biases
• Consider the human context behind the data
• Design interventions that respect student agency and circumstances

The three visualizations—bar chart, pie chart, and histogram—each tell part of
the story. Together, they provide a more complete picture than any single chart
could offer. This is the power of effective data storytelling: using the right
visualization for each message, and weaving them into a coherent narrative.

For educators and administrators, the message is clear but nuanced: while study
hours and attendance matter, supporting student success requires understanding
the root causes of low performance and removing barriers to engagement, not just
penalizing low attendance or marks.

For students, the message is empowering: there are clear factors within your
control (study approach, attendance) that correlate with success. However, if
you're struggling, poor marks probably reflect specific challenges (teaching
style mismatch, personal circumstances, prerequisite gaps) that targeted support
can address, rather than inherent inability.

This data story illustrates why data science is ultimately about enhancing human
decision-making, not replacing it.
    """)
    
    print("\n")
    print("="*80)
    print("END OF DATA STORY")
    print("="*80)
    print("\n")


def main():
    """
    Main function orchestrating the entire analysis workflow.
    
    This function:
    1. Generates synthetic student data
    2. Creates three complementary visualizations
    3. Saves visualizations as PNG file
    4. Prints comprehensive data story
    
    Returns:
        None
    """
    # Print header indicating script start
    print("\n" + "="*80)
    print("STORYTELLING WITH GRAPHS - DATA SCIENCE LEARNING ASSIGNMENT")
    print("="*80 + "\n")
    
    # Step 1: Generate the synthetic student dataset
    print("Step 1: Generating synthetic student dataset...")
    df = generate_synthetic_student_data()
    print(f"✓ Dataset created with {len(df)} student records")
    print(f"✓ Columns: {', '.join(df.columns)}\n")
    
    # Step 2: Create and save visualizations
    print("Step 2: Creating visualizations...")
    create_visualizations(df)
    print()
    
    # Step 3: Print data story
    print("Step 3: Generating data story...")
    print_data_story(df)
    
    # Print completion message
    print("="*80)
    print("ASSIGNMENT COMPLETE!")
    print("="*80)
    print("\nOutput file: 28_02_2026_Storytelling_with_Graphs.png")
    print("Data Story: Printed above (900+ words)")
    print("\n")


# Run the main function when script is executed
if __name__ == "__main__":
    main()
