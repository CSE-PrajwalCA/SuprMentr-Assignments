"""
================================================================================
LOGIC BUILDER - FIZZBUZZ IMPLEMENTATION & MODULO OPERATOR MASTERY
================================================================================

TITLE:
Logic Builder - Understanding the Modulo Operator and Algorithm Optimization

LEARNING OBJECTIVES:
1. Understand the modulo operator (%) for remainder calculations
2. Implement the classic FizzBuzz algorithm correctly
3. Use dictionaries to aggregate and count results
4. Convert results to pandas DataFrames for formatted reports
5. Analyze algorithm efficiency (O(n) time complexity)
6. Compare imperative (explicit loops) vs functional (list comprehension) approaches
7. Understand why abstraction into functions improves readability
8. Learn when to optimize and when to keep code simple
9. Use pandas for data presentation and statistical summaries
10. Recognize patterns in numbers and logic (divisibility, combinations)

ASSIGNMENT OVERVIEW:
FizzBuzz is a classic programming challenge used to teach conditionals and loops.
We generate integers 1-50, applying these rules:

- If divisible by 3: "Fizz"
- If divisible by 5: "Buzz"
- If divisible by 15: "FizzBuzz"
- Otherwise: the number itself

This teaches the MODULO OPERATOR (%), which returns remainder after division.
Example: 15 % 3 = 0 (15 divided by 3 leaves remainder 0, so divisible)
         16 % 3 = 1 (16 divided by 3 leaves remainder 1, not divisible)

We implement FizzBuzz as a function (reusable, testable), count results in a
dictionary, and display as a formatted pandas DataFrame. We then discuss:
- Why O(n) iteration is efficient
- When to choose simple code vs optimized code
- How functions improve software quality
- Performance comparison with list comprehensions

The code heavily explains modulo math, algorithm design, and code optimization.
================================================================================
"""

import pandas as pd  # For formatted DataFrame output
import timeit      # For performance measurement

# ============================================================================
# SECTION 1: MODULO OPERATOR EXPLANATION
# ============================================================================

print("\n" + "=" * 80)
print("LOGIC BUILDER - FIZZBUZZ & MODULO OPERATOR")
print("=" * 80)

print("""
--- EDUCATIONAL SECTION: THE MODULO OPERATOR (%) ---

The modulo operator returns the REMAINDER after division.

Mathematical definition:
a % b = remainder after dividing a by b

Examples:
- 15 % 3 = 0  (15 ÷ 3 = 5 with remainder 0, so 15 is divisible by 3)
- 16 % 3 = 1  (16 ÷ 3 = 5 with remainder 1, not evenly divisible)
- 17 % 5 = 2  (17 ÷ 5 = 3 with remainder 2)
- 20 % 4 = 0  (20 ÷ 4 = 5 with remainder 0, so evenly divisible)

KEY INSIGHT: If a % b == 0, then a is perfectly divisible by b.

Uses of modulo in programming:
1. DIVISIBILITY CHECKS: Is number even or odd?
   - n % 2 == 0: even
   - n % 2 == 1: odd
   
2. CYCLING PATTERNS: Repeat pattern every N elements
   - n % 7: day of week cycling (0-6 representing Sun-Sat)
   - n % 12: months cycling
   
3. HASH TABLES: Distribute items into buckets
   - hash(item) % bucket_count: which bucket to store in?
   
4. FIZZBUZZ: Check divisibility by 3 and 5

FizzBuzz uses modulo to check:
if n % 3 == 0: number is divisible by 3
if n % 5 == 0: number is divisible by 5
if n % 15 == 0: number is divisible by both (since 15 = 3 × 5)
""")

# ============================================================================
# SECTION 2: FIZZBUZZ FUNCTION IMPLEMENTATION
# ============================================================================

def fizzbuzz(n):
    """
    Apply FizzBuzz rules to a single integer.
    
    Args:
        n (int): The number to apply FizzBuzz rules to
        
    Returns:
        str: "FizzBuzz", "Fizz", "Buzz", or the number as string
        
    LOGIC EXPLANATION:
    Check conditions in specific order (most specific first):
    1. Is divisible by 15? (both 3 AND 5): return "FizzBuzz"
       - Must check 15 first! If we check 3 first, any multiple of 3 matches
       - This ensures 15-divisible numbers are "FizzBuzz" not just "Fizz"
       
    2. Is divisible by 3 only? return "Fizz"
    
    3. Is divisible by 5 only? return "Buzz"
    
    4. Otherwise: return the number
    
    EDUCATIONAL NOTE: Order matters! This demonstrates importance of
    condition ordering in if/elif/else statements.
    """
    
    # Check most specific condition first (divisible by both 3 AND 5)
    # 15 = 3 × 5, so any number divisible by 15 is divisible by both
    # We must check this FIRST, before checking 3 or 5 individually
    if n % 15 == 0:
        # n is divisible by 15 (remainder is 0)
        # Examples: 15, 30, 45
        return "FizzBuzz"
    
    # Check if divisible by 3 only (not by 5)
    # If we already checked n%15, we know n is NOT divisible by 15
    # So if n%3==0, it's divisible by 3 but NOT by 5
    elif n % 3 == 0:
        # n is divisible by 3 (and not by 5)
        # Examples: 3, 6, 9, 12, 18, 21, 24, 27, 33, 36, ...
        return "Fizz"
    
    # Check if divisible by 5 only (not by 3)
    elif n % 5 == 0:
        # n is divisible by 5 (and not by 3)
        # Examples: 5, 10, 20, 25, 35, 40, 50, ...
        return "Buzz"
    
    # Not divisible by 3 or 5
    else:
        # Return number as string
        # Examples: 1, 2, 4, 7, 8, 11, 13, 14, 16, ...
        return str(n)


# ============================================================================
# SECTION 3: MAIN FIZZBUZZ ALGORITHM (1-50)
# ============================================================================

print("\n" + "=" * 80)
print("FIZZBUZZ OUTPUT (1-50)")
print("=" * 80 + "\n")

# Dictionary to count result types
# We'll track how many of each type: Fizz, Buzz, FizzBuzz, regular numbers
fizzbuzz_counts = {
    "Fizz": 0,
    "Buzz": 0,
    "FizzBuzz": 0,
    "Numbers": 0
}

# List to store all results (for DataFrame)
# Each element is a tuple: (number, fizzbuzz_result)
results = []

# Generate FizzBuzz for 1-50
# range(1, 51) generates integers 1 through 50 (51 is exclusive)
for num in range(1, 51):
    # Apply FizzBuzz logic to current number
    result = fizzbuzz(num)
    
    # Store result as tuple
    results.append((num, result))
    
    # Count result type for statistics
    if result == "FizzBuzz":
        fizzbuzz_counts["FizzBuzz"] += 1
    elif result == "Fizz":
        fizzbuzz_counts["Fizz"] += 1
    elif result == "Buzz":
        fizzbuzz_counts["Buzz"] += 1
    else:
        # It's a number
        fizzbuzz_counts["Numbers"] += 1
    
    # Print result (formatted for readable output)
    # Print in columns (5 items per row) for compactness
    print(f"{result:>8}", end=" ")
    if (num % 5) == 0:
        # Every 5th number, print newline
        print()


# ============================================================================
# SECTION 4: RESULTS ANALYSIS & STATISTICS
# ============================================================================

print("\n\n" + "=" * 80)
print("FIZZBUZZ STATISTICS & ANALYSIS")
print("=" * 80 + "\n")

# Create pandas DataFrame from results
df_results = pd.DataFrame(results, columns=["Number", "FizzBuzz Result"])

# Display first 10 results
print("FIRST 10 RESULTS:")
print(df_results.head(10))

# Create summary statistics
print("\n" + "-" * 80)
print("COUNT SUMMARY:")
print("-" * 80)

# Create DataFrame for counts
df_counts = pd.DataFrame([fizzbuzz_counts], index=["Count"]).T
print(df_counts)

print(f"\nTotal numbers processed: {sum(fizzbuzz_counts.values())}")
print(f"Fizz percentage: {(fizzbuzz_counts['Fizz'] / 50 * 100):.1f}%")
print(f"Buzz percentage: {(fizzbuzz_counts['Buzz'] / 50 * 100):.1f}%")
print(f"FizzBuzz percentage: {(fizzbuzz_counts['FizzBuzz'] / 50 * 100):.1f}%")
print(f"Regular Numbers percentage: {(fizzbuzz_counts['Numbers'] / 50 * 100):.1f}%")


# ============================================================================
# SECTION 5: EDUCATIONAL DISCUSSION - MODULO & EFFICIENCY
# ============================================================================

print("\n" + "=" * 80)
print("MODULO OPERATOR - DEEP MATHEMATICAL EXPLANATION")
print("=" * 80)

print("""
MATHEMATICAL FOUNDATION:

Division algorithm (Euclidean division):
For any integers a and b (b ≠ 0), there exist unique integers q (quotient) and 
r (remainder) such that:
a = b*q + r, where 0 ≤ r < b

The remainder r is exactly what a % b computes.

Example: 17 % 5
- 17 = 5*3 + 2
- So 17 % 5 = 2

For FizzBuzz:
- 15 % 3: 15 = 3*5 + 0, so remainder is 0 (divisible)
- 15 % 5: 15 = 5*3 + 0, so remainder is 0 (divisible)
- 15 % 15: 15 = 15*1 + 0, so remainder is 0 (divisible)
- 16 % 3: 16 = 3*5 + 1, so remainder is 1 (not divisible)

EFFICIENCY ANALYSIS (O(1) per operation):

Modulo operation is O(1) - constant time - hardware CPU instruction
Modern CPUs can compute a % b almost as fast as addition/subtraction

FizzBuzz algorithm complexity:
- Loop: O(n) - iterates n times (1 to 50 = 50 iterations)
- Each iteration: 3 modulo operations (checking % 15, % 3, % 5) = O(1)
- Total: O(n) * O(1) = O(n)

For n=50: 50 * 3 = 150 modulo operations (microseconds)

Scaling:
- n=1,000: 3,000 operations (microseconds)
- n=1,000,000: 3,000,000 operations (milliseconds)
- n=1,000,000,000: 3,000,000,000 operations (seconds)

LINEAR SCALING: Double the input, double the operations
This is EFFICIENT because we must examine each number at least once.
""")


# ============================================================================
# SECTION 6: COMPARISON - DIFFERENT IMPLEMENTATIONS
# ============================================================================

print("\n" + "=" * 80)
print("IMPLEMENTATION COMPARISON")
print("=" * 80)

# Imperative approach (what we did above - explicit loops)
def fizzbuzz_imperative(limit):
    """Explicit loop version (readable, clear intent)."""
    results = []
    for num in range(1, limit + 1):
        if num % 15 == 0:
            results.append("FizzBuzz")
        elif num % 3 == 0:
            results.append("Fizz")
        elif num % 5 == 0:
            results.append("Buzz")
        else:
            results.append(str(num))
    return results


# Functional approach (list comprehension - Pythonic, concise)
def fizzbuzz_functional(limit):
    """List comprehension version (concise, Pythonic)."""
    # Ternary operators (condition? value_if_true : value_if_false)
    # Nested ternary for FizzBuzz logic
    return [
        "FizzBuzz" if num % 15 == 0 
        else "Fizz" if num % 3 == 0 
        else "Buzz" if num % 5 == 0 
        else str(num)
        for num in range(1, limit + 1)
    ]


# One-liner approach (compact but less readable)
def fizzbuzz_oneliner(limit):
    """One-liner using map and lambda (compact but dense)."""
    fizzbuzz_rules = lambda n: "FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)
    return list(map(fizzbuzz_rules, range(1, limit + 1)))


print("""
APPROACH 1: IMPERATIVE (For loops with explicit if/else)
Pros:
- Very readable (easy to understand intent)
- Easy to debug (can add print statements in loop)
- Allows complex logic (multiple statements per iteration)
- Preferred in professional code (other developers understand it)

Cons:
- More verbose (more lines of code)

APPROACH 2: FUNCTIONAL (List comprehension)
Pros:
- Concise (fewer lines of code)
- Pythonic (leverages Python's strengths)
- Creates list directly (no separate append calls)

Cons:
- Less readable for complex logic
- Nested ternaries get confusing

APPROACH 3: ONE-LINER (map + lambda)
Pros:
- Very compact

Cons:
- Hard to read
- Hard to debug
- Overkill for most use cases

RECOMMENDATION FOR PRODUCTION CODE:
Use Approach 1 (imperative) unless performance is critical.
Readability > cleverness. Future developers will thank you.
""")

# Performance comparison
print("\nPERFORMANCE COMPARISON (100,000 iterations):")
print("-" * 80)

# Time each approach
time_imperative = timeit.timeit(lambda: fizzbuzz_imperative(100000), number=5)
time_functional = timeit.timeit(lambda: fizzbuzz_functional(100000), number=5)
time_oneliner = timeit.timeit(lambda: fizzbuzz_oneliner(100000), number=5)

print(f"Imperative approach:  {time_imperative:.4f} seconds (over 5 runs)")
print(f"Functional approach:  {time_functional:.4f} seconds (over 5 runs)")
print(f"One-liner approach:   {time_oneliner:.4f} seconds (over 5 runs)")

# Calculate speedup (difference)
print(f"\nFunctional ~{(time_functional / time_imperative):.2f}x the speed of Imperative")
print(f"One-liner ~{(time_oneliner / time_imperative):.2f}x the speed of Imperative")

print("""
INSIGHT: All approaches have roughly same speed (O(n)).
Performance difference is minimal (usually <20%).
Write readable code. Optimize only if profiling shows bottleneck.
""")


# ============================================================================
# SECTION 7: REAL-WORLD APPLICATIONS OF MODULO
# ============================================================================

print("\n" + "=" * 80)
print("REAL-WORLD APPLICATIONS OF THE MODULO OPERATOR")
print("=" * 80)

print("""
1. EVEN/ODD DETECTION:
   if n % 2 == 0: print("Even")
   else: print("Odd")
   
2. PAGINATION:
   items_per_page = 20
   page_number = record_index // items_per_page  # Division for page
   page_offset = record_index % items_per_page   # Modulo for position within page
   
3. ROUND-ROBIN SCHEDULING:
   task_id = request_id % num_threads
   # Distribute requests evenly across threads (0 to num_threads-1)
   
4. HASH TABLES & DICTIONARIES:
   bucket_index = hash(key) % num_buckets
   # Spread keys across buckets for efficient lookup
   
5. LEAP YEAR CHECKING:
   if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
       print("Leap year")
   # Complex divisibility rules for calendar dates
   
6. CIRCULAR BUFFERS:
   next_index = (current_index + 1) % buffer_size
   # Wrap around when reaching end of buffer
   
7. BATTERY CYCLES:
   charge_cycles = total_operations % max_cycles_per_charge
   # Track how many charge cycles consumed
   
8. GAME PHYSICS:
   player_direction = player_angle % 360  # Keep angle in 0-359 degrees
   # Normalize angles to prevent overflow
""")


# ============================================================================
# SECTION 8: FINAL SUMMARY WITH FORMATTED TABLE
# ============================================================================

print("\n" + "=" * 80)
print("ALGORITHM SUMMARY")
print("=" * 80)

summary_data = {
    "Metric": ["Algorithm Complexity", "Operations per 50 numbers", "Divisibilities checked per number", "Most efficient implementation"],
    "Value": ["O(n) - Linear", "3 modulo operations × 50 = 150", "Up to 3 per number", "Imperative (readable and fast)"]
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

print("\n" + "=" * 80)
print("KEY LEARNING POINTS")
print("=" * 80)
print("""
1. The modulo operator (%) returns remainder after division
2. Use a % b == 0 to check if a is divisible by b
3. Order conditions from most-specific to least-specific (15 before 3 & 5)
4. FizzBuzz demonstrates conditional logic and loops
5. O(n) algorithm means linear time complexity (scales proportionally)
6. Readable code > clever code (choose imperative > functional for production)
7. Functions make algorithms reusable and testable
8. Dictionaries efficiently count and aggregate results
9. Pandas DataFrames provide beautiful formatted output
10. Performance matters, but readability matters more (unless bottleneck proven)
""")

# Assignment implementation placeholder
print("\n✓ Script loaded and ready for execution")
print(f"Assignment: 17_02_2026_Logic_Builder.py")
print(f"Status: Implementation complete")
