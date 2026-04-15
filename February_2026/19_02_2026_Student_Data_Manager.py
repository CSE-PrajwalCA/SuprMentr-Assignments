"""
================================================================================
STUDENT DATA MANAGER - DATA STRUCTURES & AGGREGATION OPERATIONS
================================================================================

TITLE:
Student Data Manager - Working with Lists of Dictionaries & Pandas Report Generation

LEARNING OBJECTIVES:
1. Create and manage structured data using dictionaries (key-value pairs)
2. Store multiple records in lists of dictionaries
3. Iterate through data structures with for loops
4. Perform aggregation operations (total, average, min, max)
5. Create custom functions for data analysis and summary statistics
6. Assign grades based on marks using conditional logic
7. Find top performers (filtering and sorting)
8. Use pandas DataFrame for formatted data reporting
9. Understand why properly structured data simplifies analysis
10. Learn data validation patterns (checking data integrity)

ASSIGNMENT OVERVIEW:
This program manages 5 student records (name, marks, attendance). For each
student, we calculate total marks across subjects, average percentage, and
assign grades (A, B, C, D, F). We demonstrate modular functions for analysis
and use pandas to generate beautiful formatted reports. This teaches:

1. DATA STRUCTURE: Dictionary for student (key=attribute, value=data)
2. AGGREGATION: Total marks, average percentage calculation
3. GRADING: if/elif/else rules to assign letter grades
4. RANKING: Finding top performers (max, sorted)
5. REPORTING: Pandas DataFrames for formatted output

Real-world relevance: Every database application follows this pattern.
- Web apps store user profiles (dicts in databases)
- Analytics platforms aggregate data (sum, average, max)
- Schools assign grades based on criteria
- Dashboards report on top performers (sorting/filtering)

This program emphasizes that well-structured data enables insights.
Poorly structured data (spreadsheets, files) requires manual analysis.
================================================================================
"""

import pandas as pd  # For formatted reports
from statistics import mean  # For averaging

# ============================================================================
# SECTION 1: PRINT EDUCATIONAL PREAMBLE
# ============================================================================

print("\n" + "=" * 80)
print("STUDENT DATA MANAGER - DATA AGGREGATION & ANALYSIS")
print("=" * 80)

print("""
--- EDUCATIONAL SECTION: DATA STRUCTURES IN PYTHON ---

Three main data structures:
1. LIST: Ordered collection (duplicates allowed)
   - Example: [1, 2, 3, 2, 4]
   - Access by index: list[0] = 1
   - Use: when order matters, dynamic size
   
2. DICTIONARY: Key-value pairs (no duplicates, keys unique)
   - Example: {"name": "Alice", "age": 25}
   - Access by key: dict["name"] = "Alice"
   - Use: when you need key-based lookup
   
3. LIST OF DICTIONARIES: Multiple records
   - Example: [
       {"name": "Alice", "marks": 85, "attendance": 95},
       {"name": "Bob", "marks": 78, "attendance": 88},
     ]
   - Use: database-like structure (each dict = database row)

In this program:
- Each student is a DICTIONARY (keys: name, marks, attendance)
- All students stored in a LIST (ordered collection of students)
- This mirrors how databases work (tables = list of records)

Advantages:
- Structured data (no accidental misalignment)
- Type safety (Python knows it's a dict)
- Self-documenting (keys describe data)
- Sortable/filterable (pandas operations)
""")

# ============================================================================
# SECTION 2: STUDENT DATA DEFINITION
# ============================================================================

# Create list of exactly 5 student records
# Each student is a dictionary with keys: name, marks, attendance
students = [
    {
        "name": "Alice Johnson",
        "marks": [85, 90, 88, 92],  # Marks in 4 subjects
        "attendance": 96  # Attendance percentage
    },
    {
        "name": "Bob Smith",
        "marks": [78, 82, 79, 80],
        "attendance": 88
    },
    {
        "name": "Charlie Brown",
        "marks": [92, 95, 93, 94],
        "attendance": 98
    },
    {
        "name": "Diana Prince",
        "marks": [88, 87, 89, 86],
        "attendance": 94
    },
    {
        "name": "Evan Davis",
        "marks": [72, 75, 70, 76],
        "attendance": 82
    }
]

# ============================================================================
# SECTION 3: FUNCTION DEFINITIONS FOR DATA ANALYSIS
# ============================================================================

def calculate_total(marks):
    """
    Calculate total marks across all subjects.
    
    Args:
        marks (list): List of marks in each subject
        
    Returns:
        int: Sum of all marks
        
    EDUCATIONAL NOTE:
    - sum() is built-in Python function (O(n) time)
    - Demonstrates list aggregation pattern
    """
    # sum() iterates through list and adds all elements
    # sum([85, 90, 88, 92]) = 355
    return sum(marks)


def calculate_average(marks):
    """
    Calculate average percentage of marks.
    
    Args:
        marks (list): List of marks in each subject
        
    Returns:
        float: Average percentage (0-100)
        
    EDUCATIONAL NOTE:
    - Average = Total / Count
    - Formula: (85+90+88+92) / 4 = 355 / 4 = 88.75
    - Represents typical performance across subjects
    """
    # Calculate average: sum of marks / number of subjects
    # mean() from statistics module is equivalent to sum() / len()
    return mean(marks)


def assign_grade(average_marks):
    """
    Assign letter grade based on average percentage.
    
    Args:
        average_marks (float): Average marks percentage
        
    Returns:
        str: Letter grade (A, B, C, D, F)
        
    GRADING SCALE:
    - A: 90-100 (Excellent)
    - B: 80-89  (Good)
    - C: 70-79  (Satisfactory)
    - D: 60-69  (Needs Improvement)
    - F: <60    (Fail)
    
    EDUCATIONAL NOTE:
    - Demonstrates if/elif/else for categorization
    - Order matters (check from highest to lowest)
    - Could also use dictionary lookup (more scalable for many grades)
    """
    
    # Check conditions in order from highest to lowest
    if average_marks >= 90:
        return "A"  # Excellent performance
    elif average_marks >= 80:
        return "B"  # Good performance
    elif average_marks >= 70:
        return "C"  # Satisfactory performance
    elif average_marks >= 60:
        return "D"  # Needs improvement
    else:
        return "F"  # Below passing grade


def find_topper(students_list):
    """
    Find student with highest average marks.
    
    Args:
        students_list (list): List of student dictionaries
        
    Returns:
        dict: Student dictionary with highest average
        
    EDUCATIONAL NOTE:
    - Demonstrates iteration and comparison
    - Could also use max(students_list, key=lambda s: calculate_average(s['marks']))
    - This is more explicit and easier to understand
    """
    
    # Initialize with first student
    topper = students_list[0]
    topper_average = calculate_average(topper["marks"])
    
    # Iterate through remaining students
    for student in students_list[1:]:
        # Calculate current student's average
        current_average = calculate_average(student["marks"])
        
        # If current student has higher average, they're new topper
        if current_average > topper_average:
            topper = student
            topper_average = current_average
    
    return topper


# ============================================================================
# SECTION 4: DATA ANALYSIS & REPORT GENERATION
# ============================================================================

print("\n" + "=" * 80)
print("STUDENT PERFORMANCE REPORT")
print("=" * 80 + "\n")

# Create data for DataFrame
report_data = []

# Process each student and collect analysis
for student in students:
    # Calculate statistics for this student
    total_marks = calculate_total(student["marks"])
    average_marks = calculate_average(student["marks"])
    grade = assign_grade(average_marks)
    
    # Add to report
    report_data.append({
        "Name": student["name"],
        "Total Marks": total_marks,
        "Average (%)": f"{average_marks:.2f}",
        "Grade": grade,
        "Attendance (%)": student["attendance"]
    })

# Create DataFrame for formatted output
df_report = pd.DataFrame(report_data)

# Print formatted report
print("DETAILED STUDENT REPORT:")
print(df_report.to_string(index=False))

# ============================================================================
# SECTION 5: TOP PERFORMER ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("TOP PERFORMER ANALYSIS")
print("=" * 80 + "\n")

# Find top performer
topper = find_topper(students)
topper_average = calculate_average(topper["marks"])
topper_grade = assign_grade(topper_average)

print(f"🏆 TOP PERFORMER: {topper['name']}")
print(f"   Average Marks: {topper_average:.2f}%")
print(f"   Grade: {topper_grade}")
print(f"   Attendance: {topper['attendance']}%")
print(f"   Subject Marks: {topper['marks']}")

# ============================================================================
# SECTION 6: CLASS STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("CLASS STATISTICS")
print("=" * 80 + "\n")

# Calculate class-wide statistics
all_averages = [calculate_average(s["marks"]) for s in students]

# Generate statistics summary
statistics = {
    "Metric": [
        "Highest Average",
        "Lowest Average",
        "Class Average",
        "Highest Attendance",
        "Lowest Attendance",
        "Number of A Grades",
        "Number of B Grades",
        "Number of C Grades",
    ],
    "Value": [
        f"{max(all_averages):.2f}%",
        f"{min(all_averages):.2f}%",
        f"{mean(all_averages):.2f}%",
        f"{max([s['attendance'] for s in students])}%",
        f"{min([s['attendance'] for s in students])}%",
        sum([1 for s in students if assign_grade(calculate_average(s["marks"])) == "A"]),
        sum([1 for s in students if assign_grade(calculate_average(s["marks"])) == "B"]),
        sum([1 for s in students if assign_grade(calculate_average(s["marks"])) == "C"]),
    ]
}

df_stats = pd.DataFrame(statistics)
print(df_stats.to_string(index=False))

# ============================================================================
# SECTION 7: EDUCATIONAL DISCUSSION
# ============================================================================

print("\n" + "=" * 80)
print("DATA STRUCTURE DESIGN PRINCIPLES")
print("=" * 80)

print("""
WHY THIS STRUCTURE WORKS WELL:

1. SELF-DOCUMENTING:
   Each key clearly states what data it holds
   {"name": "Alice", "marks": [85, 90, 88, 92], "attendance": 96}
   vs anonymous list: ["Alice", 85, 90, 88, 92, 96] (confusing)

2. TYPE SAFETY:
   We know marks is a list (can iterate)
   We know attendance is a number (can compare)
   
3. SCALABILITY:
   Easy to add new fields: {"name": ..., "marks": ..., "attendance": ..., "email": "..."}
   Functions still work with old structure (backward compatible)
   
4. DATABASE COMPATIBILITY:
   This structure directly maps to SQL:
   CREATE TABLE students (
       name TEXT,
       marks ARRAY,
       attendance INTEGER
   );
   
5. ANALYSIS-FRIENDLY:
   List of dicts → pandas DataFrame (one line)
   DataFrame provides: filtering, sorting, aggregation, visualization

ANTI-PATTERN (what NOT to do):

❌ Mixed structure (inconsistent):
students = [["Alice", 85, 90, 88, 92, 96],
            {"name": "Bob", "marks": ...}]  # Two different formats!

❌ Nested lists (confusing):
students = [["Alice", [85, 90, 88, 92], 96],  # Which level is what?
            ["Bob", [78, 82, 79, 80], 88]]

❌ Global variables everywhere:
alice_marks = [85, 90, 88, 92]  # Hard to scale to 1000 students
bob_marks = [78, 82, 79, 80]

BEST PRACTICE:
Structured data (dicts) > unstructured (lists)
Makes code clearer and analysis easier.
""")


# ============================================================================
# SECTION 8: GRADE DISTRIBUTION VISUALIZATION (TEXT)
# ============================================================================

print("\n" + "=" * 80)
print("GRADE DISTRIBUTION")
print("=" * 80 + "\n")

# Count grades
grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
for student in students:
    avg = calculate_average(student["marks"])
    grade = assign_grade(avg)
    grade_counts[grade] += 1

# Simple text histogram
for grade in ["A", "B", "C", "D", "F"]:
    bar = "█" * grade_counts[grade]
    print(f"Grade {grade}: {bar} ({grade_counts[grade]} students)")


# ============================================================================
# SECTION 9: INDIVIDUAL DETAILED REPORTS
# ============================================================================

print("\n" + "=" * 80)
print("INDIVIDUAL STUDENT DETAILED REPORTS")
print("=" * 80)

for i, student in enumerate(students, 1):
    print(f"\n--- STUDENT {i}: {student['name']} ---")
    print(f"Subject Marks:     {student['marks']}")
    total = calculate_total(student["marks"])
    avg = calculate_average(student["marks"])
    grade = assign_grade(avg)
    print(f"Total:             {total}")
    print(f"Average:           {avg:.2f}%")
    print(f"Grade:             {grade}")
    print(f"Attendance:        {student['attendance']}%")


print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)

print("="*80)
print("19 02 2026 STUDENT DATA MANAGER")
print("="*80)

# Assignment implementation placeholder
print("\n✓ Script loaded and ready for execution")
print(f"Assignment: 19_02_2026_Student_Data_Manager.py")
print(f"Status: Implementation complete")
