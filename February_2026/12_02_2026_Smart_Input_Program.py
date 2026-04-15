"""
================================================================================
SMART INPUT PROGRAM - INTERACTIVE USER DATA COLLECTION WITH VALIDATION
================================================================================

TITLE:
Smart Input Program - Interactive Console Application with Validation & Categorization

LEARNING OBJECTIVES:
1. Master user input handling with robust error management (try/except blocks)
2. Implement age validation with integer type checking and range constraints
3. Design control flow with if/elif/else for age categorization
4. Create reusable functions that generate personalized messages
5. Use Python's built-in string methods (.strip(), .upper()) for input cleaning
6. Understand f-string syntax for readable string interpolation
7. Design user-friendly console experiences with clear prompts and feedback
8. Implement while loops for repeated input until valid
9. Learn best practices in input validation and error handling
10. Apply principles of clean code: separation of concerns, modular functions

ASSIGNMENT OVERVIEW:
This assignment teaches proper input handling in production-quality code. We build
an interactive program that collects a user's name and age, validates rigorously,
categorizes by age bracket, and generates personalized messages. The program
demonstrates why input validation is 80% of defensive programming. We handle:

1. NAME VALIDATION: Check for empty input, enforce minimum length
2. AGE VALIDATION: Try to convert to integer, catch ValueError, range check
3. CATEGORIZATION: Use if/elif/else to assign life stage (child, teen, adult, senior)
4. PERSONALIZATION: Generate custom messages based on user age category
5. REPEATED LOOPS: Ask again if input invalid (while loop pattern)
6. USER EXPERIENCE: Clear prompts, helpful error messages, formatted output

Real-world context: Every form on the internet does this. Every mobile app. Banks
verify your age. Healthcare systems validate patient data. A single validation error
could be a security vulnerability (age validation becomes a legal issue). This
program teaches the foundations of secure input handling.

The code heavily comments every line to explain:
- Why we validate inputs (security, correctness, UX)
- How exception handling prevents crashes
- What makes a good error message
- Why functions improve code reusability
- How to format output in readable tables
================================================================================
"""

# ============================================================================
# SECTION 1: INPUT VALIDATION PRINCIPLES (Printed Educational Preamble)
# ============================================================================

print("\n" + "=" * 80)
print("SMART INPUT PROGRAM - INTERACTIVE USER DATA COLLECTION")
print("=" * 80)

print("""
--- EDUCATIONAL SECTION: WHY INPUT VALIDATION MATTERS ---

This program demonstrates the foundation of secure, user-friendly applications.
EVERY piece of user input is potentially dangerous if not validated:

1. SECURITY: Unvalidated input can lead to:
   - SQL Injection: User enters "'; DROP TABLE users; --" in age field
   - Type errors: Expected integer, got string → crash
   - Buffer overflow: Very long input → memory corruption
   - Social engineering: Phishing through misleading prompts
   
2. CORRECTNESS: Invalid data breaks your program:
   - Age of -5 years doesn't make sense
   - Age of 999999 is probably a typo
   - Empty name breaks personalized messages
   - Currency fields with text → mathematical errors
   
3. USER EXPERIENCE: Good validation makes apps easier to use:
   - Clear error messages help users understand what went wrong
   - Repeated prompts until valid (instead of crash) is considerate
   - Type conversion (string → integer) happens behind the scenes
   
4. LEGAL: Some validation is required by law:
   - Age verification (for alcohol purchases, gambling, child protection)
   - Email validation (for GDPR compliance, contact requirements)
   - Phone numbers (for 2FA systems)
   
This program practices all these principles in a simple, educational context.
""")

# ============================================================================
# SECTION 2: CONTROL FLOW EXPLANATION
# ============================================================================

print("""
--- EDUCATIONAL SECTION: CONTROL FLOW WITH IF/ELIF/ELSE ---

Control flow determines program behavior based on conditions.

Example: Age-based categorization
if age < 13:
    category = "Child"
elif age < 18:
    category = "Teenager"
elif age < 65:
    category = "Adult"
else:
    category = "Senior"

Why if/elif/else (not just if)?
- First 'if': test most restrictive condition first
- 'elif': "else if" - only evaluated if previous condition false (efficient)
- 'else': fallback (if all other conditions false)

This prevents redundant testing and makes code faster + clearer.

RULE: Order conditions from most-specific to least-specific.
If you write 'elif age < 18' before 'if age < 13', 13-year-olds match both.
By ordering specific ranges first, you avoid logic errors.
""")

# ============================================================================
# SECTION 3: F-STRING FORMATTING EXPLANATION
# ============================================================================

print("""
--- EDUCATIONAL SECTION: F-STRING SYNTAX FOR READABLE OUTPUT ---

F-strings (formatted string literals) make output readable:

BAD (harder to read):
message = "Hello " + name + ", your age is " + str(age)

OKAY (using .format()):
message = "Hello {}, your age is {}".format(name, age)

GOOD (using f-string):
message = f"Hello {name}, your age is {age}"

F-string benefits:
1. READABILITY: Variable names visible in string (no positional args)
2. EXPRESSIONS: Compute inside braces f"{age * 2}" → no temp variables
3. FORMATTING: Control decimal places f"{3.14159:.2f}" → "3.14"
4. SPEED: F-strings are faster than .format() (compiled to optimized code)

In this program, we use f-strings for:
- Age confirmation messages
- Personalized responses
- Formatted output tables

This is the MODERN Python style (Python 3.6+). Older code uses .format().
You'll see f-strings everywhere in professional code.
""")

# ============================================================================
# SECTION 4: FUNCTION DEFINITIONS WITH DOCSTRINGS
# ============================================================================

def get_valid_name():
    """
    Prompt user for name and validate it.
    
    Returns:
        str: Valid name (minimum 2 characters, non-empty)
        
    Raises:
        None (loops until valid)
        
    EDUCATIONAL NOTES:
    - Uses while True loop pattern (repeat until valid)
    - Calls .strip() to remove leading/trailing whitespace
    - Calls .upper() to standardize (first letter capitalized)
    - Returns immediately on valid input (exit loop)
    """
    
    # Keep prompting until valid input received
    # This pattern ensures we can't proceed with invalid data
    while True:
        # Prompt user with clear message
        # Indentation matters—code inside while loop repeats until break
        name = input("\n--- NAME INPUT ---\nEnter your name: ").strip()
        
        # Validation check: is name empty or too short?
        # After .strip(), leading/trailing spaces removed, so len() is accurate
        if len(name) == 0:
            # User entered only whitespace (spaces, tabs, newlines)
            print("❌ Name cannot be empty. Please try again.")
            # Loop continues, user prompted again
            continue
        
        if len(name) < 2:
            # Name is too short (e.g., "A" is valid English initial but too vague)
            # Enforce minimum 2 characters for meaningful names
            print(f"❌ Name is too short ('{name}'). Minimum 2 characters required.")
            continue
        
        if not name.replace(" ", "").isalpha():
            # Check if name contains only letters (and spaces)
            # .replace(" ", "") removes spaces, then .isalpha() checks if all letters
            # This rejects names with numbers or special characters (safety)
            print(f"❌ Name contains invalid characters ('{name}'). Use letters only.")
            continue
        
        # All validations passed—name is valid, return it
        # First letter capitalized (cosmetic improvement)
        return name.capitalize()


def get_valid_age():
    """
    Prompt user for age and validate it as integer within reasonable range.
    
    Returns:
        int: Valid age (0-150 years)
        
    Raises:
        ValueError: Caught internally, prompts user again
        
    EDUCATIONAL NOTES:
    - Demonstrates try/except for type validation
    - int() conversion is dangerous (can raise ValueError)
    - Wrapping in try/except prevents crash
    - Range checking (0-150) ensures sensible values
    - Repeated prompting is better UX than single attempt + error
    """
    
    # Keep asking until valid age received
    while True:
        try:
            # Prompt user for age input
            age_input = input("\n--- AGE INPUT ---\nEnter your age: ").strip()
            
            # CRITICAL: Try to convert string to integer
            # If user enters "abc" instead of number, int() raises ValueError
            # Without try/except, program crashes here
            age = int(age_input)
            
            # Age range validation (reasonable bounds)
            # age < 0: impossible (haven't been born in past)
            # age > 150: extremely unlikely (oldest person ever: 122 years)
            # These bounds catch typos: user enters "1999" (birth year) instead of "25" (age)
            if age < 0:
                print(f"❌ Age cannot be negative ('{age}'). Please enter valid age.")
                continue
            
            if age > 150:
                print(f"❌ Age is unrealistic ('{age}'). Maximum 150 years allowed.")
                continue
            
            # All validations passed—return valid age
            return age
        
        except ValueError:
            # User entered something that's not a number
            # int("abc") raises ValueError
            # We catch it, give helpful feedback, and loop to ask again
            print(f"❌ Age must be a number ('{age_input}' is not valid). Please try again.")
            # Loop continues, user prompted again


def get_age_category(age):
    """
    Categorize age into life stage brackets.
    
    Args:
        age (int): User's age
        
    Returns:
        str: Age category (Child, Teenager, Adult, Senior)
        
    EDUCATIONAL NOTES:
    - Uses if/elif/else for exclusive categories
    - Categories must be non-overlapping and exhaustive
    (every age 0-150 falls into exactly one category)
    - Ranges ordered from most restrictive to least
    """
    
    # Category boundaries (non-overlapping):
    # 0-12 = Child, 13-19 = Teenager, 20-64 = Adult, 65+ = Senior
    if age < 13:
        # Early childhood through late childhood (0-12 years)
        return "Child"
    
    elif age < 20:
        # Teen years (13-19 years)
        # Note: Could use <= 19 or < 20; both are equivalent
        return "Teenager"
    
    elif age < 65:
        # Working-age adult (20-64 years)
        return "Adult"
    
    else:
        # Retirement age and beyond (65+ years)
        # This handles all remaining ages (65+)
        return "Senior"


def generate_personalized_message(name, age, category):
    """
    Generate personalized message based on user's age category.
    
    Args:
        name (str): User's name
        age (int): User's age
        category (str): Age category (Child, Teenager, Adult, Senior)
        
    Returns:
        str: Personalized message for the user
        
    EDUCATIONAL NOTES:
    - Different message for each category
    - Uses f-strings for dynamic content
    - Shows how functions reduce code duplication
    (instead of copy-paste message generation, single function handles all)
    """
    
    # Dictionary of messages by category
    # Key = category, Value = message template
    messages = {
        "Child": f"{name}, you're {age} years old! 🎉 Enjoy your childhood adventure. Have fun learning and playing!",
        "Teenager": f"{name}, age {age}! You're in your teenage years. 🚀 This is a time of discovery and growth!",
        "Adult": f"{name}, at age {age}, you're in your prime adult years! 💼 Keep striving for your goals!",
        "Senior": f"{name}, wisdom comes with age {age}! 🌟 Enjoy the fruits of your life's experience!"
    }
    
    # Return the appropriate message for this user's category
    # .get(category) retrieves the message; if key not found, returns None (safe)
    return messages.get(category, "Welcome!")


def print_summary_report(name, age, category):
    """
    Print a formatted summary report of user data.
    
    Args:
        name (str): User's name
        age (int): User's age
        category (str): Age category
        
    EDUCATIONAL NOTES:
    - Demonstrates formatted output (boxes, tables, alignment)
    - Separates data collection from presentation (clean code)
    - Multiple print statements create readable output with visual hierarchy
    """
    
    # Print formatted box around summary
    print("\n" + "=" * 60)
    print("USER DATA SUMMARY")
    print("=" * 60)
    
    # Print each field with clear labeling and formatting
    # Using f-strings for alignment
    print(f"Name:       {name:.<40}")  # Left-aligned, padded with dots
    print(f"Age:        {age} years")
    print(f"Category:   {category}")
    
    print("=" * 60)


# ============================================================================
# SECTION 5: MAIN PROGRAM FLOW
# ============================================================================

def main():
    """
    Main program: orchestrates input collection and personalized response.
    
    EDUCATIONAL NOTES:
    - main() function is good practice (single entry point)
    - Calls other functions in sequence (modular design)
    - Each function has single responsibility (separation of concerns)
    """
    
    # Step 1: Get valid name from user
    # get_valid_name() loops until valid, returns name string
    name = get_valid_name()
    print(f"✓ Name accepted: {name}")
    
    # Step 2: Get valid age from user
    # get_valid_age() loops until valid, returns age integer
    age = get_valid_age()
    print(f"✓ Age accepted: {age}")
    
    # Step 3: Determine age category
    # Based on age, determine which bracket user falls into
    category = get_age_category(age)
    
    # Step 4: Generate personalized message
    # Message tailored to user's age category
    message = generate_personalized_message(name, age, category)
    
    # Step 5: Print summary and personalized message
    print_summary_report(name, age, category)
    print(f"\n{message}\n")
    
    # Step 6: Print educational analysis
    print("\n" + "=" * 60)
    print("VALIDATION ANALYSIS")
    print("=" * 60)
    print(f"""
1. NAME VALIDATION: ✓
   - Length checked (minimum 2 characters)
   - Content validated (letters + spaces only)
   - Input cleaned (.strip() removes extra spaces)
   
2. AGE VALIDATION: ✓
   - Type checked (converted to integer)
   - ValueError caught if non-numeric input
   - Range checked (0-150 years is reasonable)
   
3. CATEGORIZATION: ✓
   - Age {age} falls into category: {category}
   - Used if/elif/else for non-overlapping ranges
   - Each user matches exactly one category
   
4. PERSONALIZATION: ✓
   - Message generated based on category
   - f-string interpolation used for dynamic content
   - User sees their name and age in response
   
This program demonstrates defensive programming best practices:
- Validate all inputs before use
- Provide clear error messages
- Loop until valid (don't crash on bad input)
- Separate concerns (functions do one thing well)
- Use formatted output for readability
""")


# ============================================================================
# SECTION 6: PRINTING EDUCATIONAL SECTIONS BEFORE EXECUTION
# ============================================================================

print("""
--- EDUCATIONAL SECTION: ERROR HANDLING WITH TRY/EXCEPT ---

Try/except is your shield against crashes:

try:
    # Risky operation that might raise an exception
    age = int("abc")  # This raises ValueError
except ValueError:
    # Catch the exception, prevent crash, handle gracefully
    print("That's not a number!")

Common exceptions in input validation:
- ValueError: User input can't be converted to expected type
- IndexError: Accessing list element that doesn't exist
- KeyError: Dictionary key doesn't exist
- ZeroDivisionError: Dividing by zero

Without try/except, any of these crashes the program.
With try/except, program continues, asks user to try again.
This is 100x better UX.

BEST PRACTICE:
Never let user input crash your program. Always validate.
Always provide feedback on why input was rejected.
Always ask again for input (don't give up).
""")

print("""
--- EDUCATIONAL SECTION: WHILE LOOPS FOR REPEATED INPUT ---

While loops repeat code until a condition becomes false:

while True:
    user_input = input("Enter age: ")
    if valid(user_input):
        break  # Exit loop, proceed
    else:
        print("Invalid input, try again")  # Loop continues

Why while True with break?
- More flexible than while condition (condition might be complex)
- Loops forever until you explicitly break out
- Easier to put break in middle of code than manage complex condition

PATTERN (used throughout this program):
while True:
    try:
        user_input = get_and_validate_input()
        return user_input  # Exit function, loop doesn't continue
    except:
        print("Invalid, try again")  # Loop continues, prompt again

This pattern ensures we can't proceed until user input is valid.
""")

# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the main program
    main()
