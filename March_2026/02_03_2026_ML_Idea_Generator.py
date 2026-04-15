"""
===============================================================================
ML AND DATA SCIENCE IDEA GENERATOR - REAL-WORLD PROBLEM IDENTIFICATION
===============================================================================

TITLE: Machine Learning Problem Generator for Multiple Domains

LEARNING OBJECTIVES:
1. Understand how to identify ML opportunities in different industry domains
2. Learn to map business problems to ML algorithm categories
3. Recognize features and labels in real-world scenarios
4. Calculate dataset requirements for ML projects
5. Identify appropriate evaluation metrics for different problem types
6. Understand the domain-specific challenges in ML deployment
7. Learn to articulate ML solutions to stakeholders
8. Recognize ethical considerations across domains
9. Understand cost-benefit analysis for ML implementation
10. Map business value to technical AI solutions

ASSIGNMENT OVERVIEW:
This assignment generates 15 machine learning problems across 3 different 
industry domains: college/education, healthcare/medicine, and shopping/retail. 
For each problem, you will:

1. Define the business problem clearly
2. Identify input features (data you need)
3. Define target output (what to predict)
4. Select appropriate ML algorithm with math explanation
5. Estimate required dataset size
6. Choose evaluation metric based on problem type
7. Discuss deployment challenges
8. Provide financial impact estimate

Each problem includes detailed explanations covering the mathematical 
foundations of the chosen algorithm, real-world implementation challenges, 
ethical considerations, and ROI calculations. This teaches you how to translate 
business requirements into ML solutions.

DOMAINS:
- College/Education (5 problems)
- Healthcare/Medicine (5 problems)
- Shopping/Retail (5 problems)
===============================================================================
"""

import numpy as np  # For reproducibility seed
import sys  # For system utilities

# ============================================================================
# Set random seed for reproducibility
# ============================================================================
np.random.seed(42)

# ============================================================================
# COLLEGE/EDUCATION PROBLEMS
# ============================================================================

def print_college_problem_1():
    """Student Success Prediction System"""
    print("\n" + "="*70)
    print("COLLEGE PROBLEM #1: Student Success Prediction")
    print("="*70)
    
    # PROBLEM DEFINITION
    print("\n[PROBLEM DEFINITION]")
    print("Colleges lose money on students who drop out. We need to predict")
    print("which first-year students are at risk of failing or dropping out.")
    print("This allows early intervention programs to target at-risk students.")
    
    # FEATURES (INPUT DATA)
    print("\n[FEATURES - What Data You Collect]")
    print("  • High school GPA (continuous, 0-4.0)")
    print("  • SAT/ACT scores (continuous, 0-1600)")
    print("  • Family income level (categorical: low/medium/high)")
    print("  • First-semester attendance % (continuous, 0-100)")
    print("  • First-semester exam scores (continuous, 0-100)")
    print("  • Library visits per week (continuous, 0-20)")
    print("  • Hours spent in study groups (continuous, 0-40)")
    print("  • Student engagement score (categorical: low/medium/high)")
    print("  • Prior math coursework (categorical: yes/no)")
    print("  • Age at enrollment (continuous, 17-60)")
    
    # TARGET OUTPUT
    print("\n[TARGET/OUTPUT - What to Predict]")
    print("  Binary classification: Will student succeed (1) or fail/dropout (0)?")
    print("  - 'Success' = cumulative GPA ≥ 2.0 after first year")
    print("  - 'Failure/Dropout' = cumulative GPA < 2.0 OR officially withdrew")
    
    # ALGORITHM & MATH
    print("\n[ALGORITHM: Logistic Regression]")
    print("  Mathematical Foundation:")
    print("    • Probability model: P(success) = 1 / (1 + e^(-z))")
    print("    • where z = β₀ + β₁*gpa + β₂*sat + β₃*engagement + ...")
    print("    • β parameters learned from historical student data")
    print("    • Output: probability between 0 and 1")
    print("  Why this algorithm?")
    print("    - Interpretable: coefficients show which features matter most")
    print("    - Fast training: works with thousands of students")
    print("    - Probabilistic: gives confidence scores for predictions")
    print("    - Generalizes well: avoids overfitting with proper regularization")
    
    # DATASET SIZE
    print("\n[DATASET SIZE REQUIREMENTS]")
    print("  • Minimum: 500 historical student records")
    print("  • Recommended: 5,000-10,000 records (multiple years of data)")
    print("  • Class balance: ~30% failures, 70% successes (realistic)")
    print("  • Data collection: 3-5 years of historical student data")
    
    # EVALUATION METRIC
    print("\n[EVALUATION METRICS]")
    print("  Primary Metric: Precision (False Positive Cost)")
    print("    - Formula: TP / (TP + FP)")
    print("    - Why: Incorrectly flagging successful students wastes intervention ($)")
    print("    - Goal: Identify true at-risk students, minimize false alarms")
    print("  Secondary Metrics:")
    print("    - Recall: Catch all at-risk students (TP / (TP + FN))")
    print("    - ROC-AUC: Threshold-independent performance (0-1 scale)")
    print("    - Calibration: Probability estimates should be reliable")
    
    # DEPLOYMENT CHALLENGES
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Data privacy: FERPA regulations (US) protect student records")
    print("  2. Feedback loops: Interventions change outcomes, bias future predictions")
    print("  3. Fairness: Model must not discriminate by race/income/family background")
    print("  4. Integration: Must connect with student information systems")
    print("  5. Transparency: Advisors need to understand WHY a student is flagged")
    
    # FINANCIAL IMPACT
    print("\n[FINANCIAL IMPACT]")
    print("  Costs:")
    print("    - Data collection & cleaning: $30,000")
    print("    - Model development: $50,000 (engineer time)")
    print("    - Intervention program: $10,000 per at-risk student identified")
    print("  Benefits:")
    print("    - Tuition retention: $50,000 per student who stays (4 semesters)")
    print("    - Reputation: 10% decrease in attrition = $500k+ revenue")
    print("    - 100-student intervention @ 70% success rate = $3.5M retained revenue")
    print("  ROI: 7:1 (for every $1 spent, $7 in tuition revenue retained)")


def print_college_problem_2():
    """Course Recommendation Engine"""
    print("\n" + "="*70)
    print("COLLEGE PROBLEM #2: Intelligent Course Recommendation System")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Students struggle to choose relevant courses. Departments want to")
    print("recommend courses that (1) match student interests, (2) align with")
    print("degree requirements, and (3) avoid schedule conflicts.")
    
    print("\n[FEATURES]")
    print("  • Student's completed courses (vector: one-hot encoding)")
    print("  • Grades received (A/B/C/D/F)")
    print("  • Student major/minor (categorical)")
    print("  • Course difficulty level (categorical: intro/intermediate/advanced)")
    print("  • Course description embedding (NLP feature from course title)")
    print("  • Prerequisites met (binary: yes/no)")
    print("  • Semester standing (year 1/2/3/4)")
    print("  • Time constraints (available slots: boolean for each day/time)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Ranked list of top-5 recommended courses with scores.")
    print("  Format: [(Course_ID, Score_0.95), (Course_ID, Score_0.87), ...]")
    
    print("\n[ALGORITHM: Collaborative Filtering + Content-Based Filtering]")
    print("  Mathematical Foundation:")
    print("    • User-item matrix: students vs courses, entries = grade received")
    print("    • Similarity: cosine(Student_A, Student_B) = dot product / norms")
    print("    • Prediction: predicted_grade = Σ(similar_students' grades) / N")
    print("    • Hybrid: combine with content similarity (course features)")
    print("  Why hybrid?")
    print("    - Collaborative: finds students with similar histories")
    print("    - Content-based: matches student profile to course features")
    print("    - Addresses cold-start: new courses still recommended via content")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 2,000 students × 200 courses = 400k enrollments")
    print("  • Historical data: 4+ years (capture curriculum evolution)")
    print("  • Density: ~5% filled (average student takes 40 courses, 800 available)")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Click-Through Rate (CTR)")
    print("    - Users who click recommended course / total recommendations")
    print("    - Real-world metric of recommendation relevance")
    print("  Secondary:")
    print("    - Precision@5: How many of top-5 are actually taken")
    print("    - Diversity: Avoid recommending only within same department")
    print("    - Serendipity: Recommendations users wouldn't discover themselves")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Cold-start for new students (no history): Use content-based")
    print("  2. Curriculum changes: Retrain quarterly with new courses")
    print("  3. Silo-breaking: Recommend across departments (requires validation)")
    print("  4. Privacy: Don't expose student grade distributions")
    print("  5. Explainability: Why was THIS course recommended to THIS student?")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Lower costs: Reduces academic advising hours by ~20%")
    print("  • Better retention: Students in recommended courses have 8% better grades")
    print("  • Cross-registration: 15% of students discover new departments")
    print("  • Savings: 10 fewer academic advisors needed ($500k annually)")
    print("  • Revenue: Inter-departmental enrollments increase majors (+$1M/year)")


def print_college_problem_3():
    """Fraudulent Financial Aid Detection"""
    print("\n" + "="*70)
    print("COLLEGE PROBLEM #3: Financial Aid Fraud Detection System")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Colleges disburse billions in financial aid. Some students commit fraud:")
    print("false income declarations, forged documents, or filing aid multiple times.")
    print("Detecting fraud saves money and maintains system integrity.")
    
    print("\n[FEATURES]")
    print("  • Reported family income (continuous, $0-$500k)")
    print("  • FAFSA data discrepancies (count of inconsistencies)")
    print("  • Tax return e-file date vs FAFSA submission date (days)")
    print("  • Number of schools applied in same year (count)")
    print("  • Sibling's aid disbursement pattern (anomaly score)")
    print("  • Student's zip code vs reported address (match: yes/no)")
    print("  • Document submission speed (hours from request to upload)")
    print("  • Previous fraud history (binary: yes/no)")
    print("  • Unusual enrollment pattern (full-time→part-time jump)")
    print("  • Aid amount vs. expected cost (ratio)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Binary classification: Fraud (1) or Legitimate (0)")
    
    print("\n[ALGORITHM: Gradient Boosting (XGBoost)]")
    print("  Mathematical Foundation:")
    print("    • Ensemble of decision trees: sequential trees correct prior errors")
    print("    • Gradient descent: minimize loss function iteratively")
    print("    • Final prediction: f(x) = Σ(tree₁ + tree₂ + ... + treeₙ)")
    print("    • Regularization: shrinkage parameter λ prevents overfitting")
    print("  Why XGBoost?")
    print("    - Handles class imbalance (fraud is rare ~0.5%)")
    print("    - Feature importance: shows which fraud signals matter")
    print("    - Captures non-linear patterns (e.g., threshold effects)")
    print("    - Fast inference: milliseconds per prediction")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 100,000 aid applications")
    print("  • Confirmed fraud cases: 500-1,000 manually reviewed cases")
    print("  • Class balance: ~0.5-1% fraud (highly imbalanced)")
    print("  • Time period: 3-5 years historical data")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Recall at 95% Precision")
    print("    - Catch fraud without false-alarming legitimate applicants")
    print("    - Formula: TP / (TP + FN)")
    print("  Secondary:")
    print("    - ROC-AUC: Performance across all thresholds")
    print("    - Precision-Recall: Handles class imbalance better")
    print("    - Cost-sensitive accuracy: weight fraud catches higher")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Label scarcity: Only ~100 confirmed fraud cases (hard to learn)")
    print("  2. Adversarial: Fraudsters adapt to detection methods")
    print("  3. False positives: Incorrectly flag legitimate students")
    print("  4. Privacy: Legal constraints on using sensitive financial data")
    print("  5. Fairness: Model must not discriminate against low-income students")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Fraud cost: ~$150,000 per school per year")
    print("  • CMS fraud data: US student aid fraud ~$400M annually")
    print("  • Detection savings: Stopping 1 fraudster saves ~$50k per school")
    print("  • Model cost: $200,000 (development + maintenance year 1)")
    print("  • ROI: Within 1-2 years, saves $500k+ per school")


def print_college_problem_4():
    """Classroom Engagement Prediction"""
    print("\n" + "="*70)
    print("COLLEGE PROBLEM #4: Real-Time Classroom Engagement Prediction")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Professors want to know: Are my students actually engaged in lectures?")
    print("High disengagement predicts lower exam scores. Early detection allows")
    print("instructors to adjust teaching methods mid-semester.")
    
    print("\n[FEATURES]")
    print("  • Attendance rate (percentage, 0-100%)")
    print("  • Classroom participation count (# questions/comments per hour)")
    print("  • Assignment submission timeliness (avg days before deadline)")
    print("  • Online quiz performance (%, 0-100)")
    print("  • Response time to email (hours, 0-168)")
    print("  • Canvas/LMS activity (logins per week, 0-50)")
    print("  • Grade trend (improving vs declining slope)")
    print("  • Office hour visits (count per semester)")
    print("  • Study group formation (yes/no, peer data)")
    print("  • Peer network (# study partners, collaboration signals)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Engagement score: 0-100 (continuous), segmented into:")
    print("    - Highly Engaged: 80-100")
    print("    - Moderately Engaged: 60-79")
    print("    - Low Engagement: <60 (warning zone)")
    
    print("\n[ALGORITHM: Random Forest Regression]")
    print("  Mathematical Foundation:")
    print("    • N bootstrap samples, train N decision trees independently")
    print("    • Each tree learns different patterns (variance reduction)")
    print("    • Final prediction: average of N tree predictions")
    print("    • Out-of-bag error: built-in cross-validation metric")
    print("  Why Random Forest?")
    print("    - Robust: resistant to outliers and noise")
    print("    - Feature importance: ranks which signals matter most")
    print("    - Non-parametric: no assumptions about data distribution")
    print("    - Multi-class capable: can output continuous engagement score")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 5,000 student-course records")
    print("  • Source: 500 courses × 10 students average over 1 year")
    print("  • Important: Needs exam data to validate engagement→performance link")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Correlation with exam performance")
    print("    - correlation(predicted_engagement, exam_score) should be >0.7")
    print("  Secondary:")
    print("    - RMSE: prediction accuracy within ±5 points")
    print("    - Precision of 'at-risk' category: who actually needs intervention")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Measurement bias: Active students may have internet issues")
    print("  2. Privacy: Monitoring student behavior raises ethical questions")
    print("  3. Causality: Low grades cause disengagement (reverse causation)")
    print("  4. Intervention timing: When to alert instructor (too many false alarms)")
    print("  5. Generalization: Model trained on intro courses may not work for seminars")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Development: $100,000 (engineer time)")
    print("  • Institutional gain: Earlier interventions reduce fail rate ~3%")
    print("  • Value: 3% of 20,000 students = 600 students")
    print("  • Savings per student: ~$50,000 retention value")
    print("  • Total: $30M+ in prevented attrition over 5 years")


def print_college_problem_5():
    """Alumni Donation Prediction"""
    print("\n" + "="*70)
    print("COLLEGE PROBLEM #5: Alumni Donation Propensity Model")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Colleges rely on donations. Not all alumni are equally likely to donate.")
    print("Predicting who will donate lets development offices target fundraising")
    print("efforts efficiently, increasing ROI of engagement campaigns.")
    
    print("\n[FEATURES]")
    print("  • Years since graduation (continuous, 1-60)")
    print("  • Undergraduate GPA (continuous, 0-4.0)")
    print("  • Graduate degree earned (categorical: none/masters/phd)")
    print("  • Student athlete (binary: yes/no)")
    print("  • Fraternity/sorority member (binary: yes/no)")
    print("  • Donor family member (binary: yes/no)")
    print("  • Employer type (categorical: private/public/nonprofit/education)")
    print("  • Estimated salary (continuous, based on alumni DB)")
    print("  • Prior donation history (binary, count, amount)")
    print("  • Homecoming attendance (count in past 5 years)")
    print("  • Alumni event participation (count, 0-10)")
    print("  • Geographic proximity (distance in miles from campus)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Binary classification: Will donate in next 2 years (1) or not (0)")
    print("  Alternative: Regression → predicted donation amount ($)")
    
    print("\n[ALGORITHM: Logistic Regression with L1 Regularization]")
    print("  Mathematical Foundation:")
    print("    • P(donate) = 1 / (1 + e^(-z))")
    print("    • Loss: cross-entropy with L1 penalty on coefficients")
    print("    • L1 penalty: encourages sparse solutions (zero out weak predictors)")
    print("    • Learned: which 3-5 features are crucial for donation prediction")
    print("  Why this algorithm?")
    print("    - Interpretable: explain to development office why someone is targeted")
    print("    - Sparse: identifies key predictor variables")
    print("    - Stable: L1 regularization prevents overfitting")
    print("    - Probabilistic: confidence in donation likelihood")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 20,000 alumni records with donation history")
    print("  • Timeframe: 5-10 years of tracked donations")
    print("  • Label: donation made within 24 months (positive) or not")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Precision at 80% recall")
    print("    - Catch 80% of donors, with high precision (few false positives)")
    print("  Secondary:")
    print("    - Lift: How much better than random targeting")
    print("    - ROI: Revenue generated / cost of campaign")
    print("    - Segment profitability: Which alumni groups most profitable")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Confirmation bias: Already-likely donors get more attention (feedback)")
    print("  2. Privacy: Alumni may resist being 'scored' for donation likelihood")
    print("  3. Fairness: Model shouldn't discriminate by wealth or gender")
    print("  4. Causation: Does targeting create donation or just select for it?")
    print("  5. Long feedback loops: Takes 2 years to validate donation behavior")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Average donation: $5,000")
    print("  • Current targeting: Random outreach, ~2% response rate")
    print("  • Model improvement: 5% response rate (2.5x lift)")
    print("  • Cost: 1000 targeted campaigns × $50 = $50,000")
    print("  • Revenue gain: 1000 × 5% × $5,000 = $250,000")
    print("  • ROI: 5:1 (for each $1 spent, $5 in donations)")


# ============================================================================
# HEALTHCARE/MEDICINE PROBLEMS
# ============================================================================

def print_healthcare_problem_1():
    """Patient Readmission Prediction"""
    print("\n" + "="*70)
    print("HEALTHCARE PROBLEM #1: Hospital Readmission Risk Prediction")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("When patients are discharged from hospital, some return within 30 days.")
    print("Readmissions cost hospitals $15,000+ per patient and indicate poor care.")
    print("Predicting readmission risk lets doctors arrange preventive follow-ups.")
    
    print("\n[FEATURES]")
    print("  • Patient age (continuous, 0-100)")
    print("  • Length of stay (days, 0-365)")
    print("  • Primary diagnosis (categorical: ICD-10 code)")
    print("  • Comorbidities (count: diabetes, hypertension, etc.)")
    print("  • Number of medications at discharge (0-20)")
    print("  • Lab values (glucose, cholesterol, hemoglobin)")
    print("  • Vital signs (blood pressure, heart rate, temperature)")
    print("  • Functional capacity at discharge (mobility: independent/assistance/dependent)")
    print("  • Living situation (categorical: home/assisted/nursing home)")
    print("  • Social support (binary: family available for care)")
    print("  • Insurance type (categorical: Medicare/commercial/uninsured)")
    print("  • Prior hospitalizations (count, last 12 months)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Binary classification: Readmitted within 30 days (1) or not (0)")
    
    print("\n[ALGORITHM: Gradient Boosting (LightGBM)]")
    print("  Mathematical Foundation:")
    print("    • Sequential decision trees: each new tree focuses on prior errors")
    print("    • Leaf-wise growth: deeper, more focused tree structure")
    print("    • Regularization: controls depth and leaf count to prevent overfitting")
    print("    • Multinomial loss: extends to multi-class problems")
    print("  Why LightGBM?")
    print("    - Fast training: handles hospital-scale data (millions of records)")
    print("    - Handles missing data: many patient records incomplete")
    print("    - Non-linear: disease interactions captured automatically")
    print("    - Feature importance: clinical interpretability")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 50,000 hospital discharge records")
    print("  • Timeframe: 3-5 years of data")
    print("  • Readmission rate: ~15-20% baseline (label distribution)")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Sensitivity (Recall) at 90% Specificity")
    print("    - Catch 90% of high-risk patients")
    print("    - Minimize false positives (expensive unnecessary interventions)")
    print("  Secondary:")
    print("    - Net Benefit: (TP/N) - (FP/N) × cost_ratio")
    print("    - Decision curve: plot benefit across all thresholds")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Regulatory: HIPAA compliance for patient data")
    print("  2. Liability: If model misses high-risk patient, hospital liable")
    print("  3. Fairness: Model must not bias against protected groups")
    print("  4. Implementation: Integration with EHR systems")
    print("  5. Physician resistance: Doctors may distrust algorithm decisions")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Readmission cost: $15,000 per patient")
    print("  • Hospital size: 500 discharges/month = 6,000/year")
    print("  • Current readmit rate: 15% = 900 readmissions/year = $13.5M")
    print("  • Target reduction: 3% (from 15% to 12%)")
    print("  • Savings: 180 readmissions prevented = $2.7M/year")
    print("  • Model cost: $300,000 development + $50k/year maintenance")
    print("  • ROI: +$2.4M net benefit year 1, ongoing")


def print_healthcare_problem_2():
    """Disease Diagnosis Prediction"""
    print("\n" + "="*70)
    print("HEALTHCARE PROBLEM #2: Automated Disease Diagnosis from Symptoms")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Patient reports symptoms; doctor diagnoses disease. In many cases,")
    print("diagnosis is uncertain or requires specialist. ML can suggest diagnoses,")
    print("helping doctors consider differential diagnoses more systematically.")
    
    print("\n[FEATURES]")
    print("  • Chief complaint (text: 'headache', 'chest pain', etc.)")
    print("  • Symptom list (categorical: fever, cough, nausea, etc.)")
    print("  • Symptom duration (days)")
    print("  • Severity (categorical: mild/moderate/severe)")
    print("  • Associated symptoms (binary features per symptom)")
    print("  • Medical history (categorical: diabetes, asthma, etc.)")
    print("  • Medications (list of current medications)")
    print("  • Allergies (categorical)")
    print("  • Age (continuous)")
    print("  • Gender (binary/categorical)")
    print("  • Recent travel (categorical: country/region)")
    print("  • Lab results (if available: glucose, WBC, etc.)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Multi-class classification: probability of each disease diagnosis")
    print("  Output: Top-3 disease probabilities (e.g., 40% flu, 30% cold, 20% RSV)")
    
    print("\n[ALGORITHM: Neural Network (Deep Learning)]")
    print("  Mathematical Foundation:")
    print("    • Layer 1: 64 neurons, ReLU activation")
    print("    • Layer 2: 32 neurons, ReLU activation")
    print("    • Output: 30 neurons (# of diseases), softmax activation")
    print("    • Loss: categorical cross-entropy")
    print("    • Optimization: Adam algorithm (adaptive learning rates)")
    print("  Why Deep Learning?")
    print("    - Captures complex symptom interactions (non-linear)")
    print("    - Handles variable-length input (different # of symptoms)")
    print("    - Multi-output: predicts disease probability distribution")
    print("    - Learned representations: discovers symptom patterns")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 100,000 patient visits with confirmed diagnosis")
    print("  • Balance: sample to represent disease prevalence")
    print("  • Validation: 10,000 cases confirmed by specialist review")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Top-3 Accuracy (correct diagnosis in top 3 predictions)")
    print("    - Clinical standard: doctor uses it as one input among many")
    print("  Secondary:")
    print("    - Per-disease sensitivity: minimize false negatives for serious diseases")
    print("    - Specificity: don't over-diagnose rare diseases")
    print("    - Calibration: predicted probability matches actual diagnosis rate")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Regulatory: FDA approval required for diagnostic devices")
    print("  2. Liability: Misdiagnosis can lead to patient harm and lawsuits")
    print("  3. Data bias: Training data from major hospitals (may not generalize)")
    print("  4. Rarity: Some diseases rare, model can't learn well from few examples")
    print("  5. Explainability: Doctors need to understand WHY diagnosis suggested")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Missed diagnoses cost: $10M+ annually in malpractice")
    print("  • Clinical decision support value: 10% reduction in diagnostic errors")
    print("  • Hospital scale: 1M patient visits/year → 10k diagnostic errors prevented")
    print("  • Savings: 10k × $50k average error cost = $500M potential value")
    print("  • But: Regulatory approval $5-10M, clinical validation required")


def print_healthcare_problem_3():
    """Treatment Response Prediction"""
    print("\n" + "="*70)
    print("HEALTHCARE PROBLEM #3: Drug Treatment Response Prediction")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Different patients respond differently to medicines. Some drugs work well,")
    print("others cause side effects. Predicting treatment response helps doctors")
    print("choose the right drug for the right patient (personalized medicine).")
    
    print("\n[FEATURES]")
    print("  • Patient genetics (SNPs in relevant genes, categorical)")
    print("  • Age (continuous)")
    print("  • Gender (binary)")
    print("  • Weight/BMI (continuous)")
    print("  • Kidney function (estimated GFR, continuous)")
    print("  • Liver function (ALT/AST, continuous)")
    print("  • Current medications (drug interactions, categorical)")
    print("  • Disease severity (continuous scale)")
    print("  • Comorbidities (binary features)")
    print("  • Previous medication responses (categorical: worked/didn't work/unknown)")
    print("  • Smoking status (categorical: current/former/never)")
    print("  • Alcohol use (categorical)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Multi-class: {Good_Response, Moderate_Response, Poor_Response}")
    print("  Or continuous: Probability of good response (0-1)")
    
    print("\n[ALGORITHM: Random Forest Classification]")
    print("  Mathematical Foundation:")
    print("    • Each tree: independent random sample of data and features")
    print("    • Ensemble: votes determine final class prediction")
    print("    • OOB estimate: built-in validation without test set")
    print("    • Feature importance: Gini/entropy decrease per feature")
    print("  Why Random Forest?")
    print("    - Handles mixed data types (genetic + clinical)")
    print("    - Robust: resistant to individual outliers")
    print("    - Feature importance: identifies key genetic/clinical factors")
    print("    - Non-parametric: no assumptions about data distribution")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 5,000 patient-treatment pairs")
    print("  • Genetic data: sequencing costs $1,000-5,000 per patient")
    print("  • Drug trials: typically 100-1,000 patients per drug")
    print("  • Timeframe: 5+ years to collect response outcomes")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Precision of 'Good Response' class")
    print("    - If we predict good response, how often correct? Goal: >80%")
    print("  Secondary:")
    print("    - Sensitivity: identify all truly good-responders")
    print("    - Clinical utility: reduce adverse events by 20%+")
    print("    - Cost-effectiveness: treatment cost × response probability")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Data scarcity: Genetic data expensive to collect (funding constraint)")
    print("  2. Regulatory: Drug response not FDA guideline (yet)")
    print("  3. Privacy: Genetic information extremely sensitive")
    print("  4. Ethical: Deny treatment to predicted low-responders? (access equity)")
    print("  5. Validation: Requires long-term patient follow-up (5+ years)")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Adverse drug reactions: 100k deaths/year in US, $100B cost")
    print("  • Precision medicine potential: 10% reduction in ADRs = $10B value")
    print("  • Individual case: $100k treatment cost avoided if predicted to fail")
    print("  • Pharma value: 3% improvement in drug success rates")
    print("  • Model development: $2-5M (gene sequencing + trials expensive)")


def print_healthcare_problem_4():
    """Medical Image Abnormality Detection"""
    print("\n" + "="*70)
    print("HEALTHCARE PROBLEM #4: Automated Medical Image Analysis")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Radiologists analyze 1000+ medical images daily. Detecting abnormalities")
    print("(tumors, fractures, infections) is repetitive and error-prone. AI can")
    print("flag suspicious regions, helping radiologists prioritize cases.")
    
    print("\n[FEATURES]")
    print("  • X-ray image pixels (2D array, 512×512 pixels, grayscale)")
    print("  • CT scan images (3D array, 512×512×100 slices)")
    print("  • MRI images (3D array, multiple weightings: T1, T2, FLAIR)")
    print("  • Patient metadata (age, sex, clinical indication)")
    print("  • Prior images (to detect change over time)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Binary: Abnormality present (1) or normal (0)")
    print("  Regional: Bounding box or segmentation mask of abnormal regions")
    
    print("\n[ALGORITHM: Convolutional Neural Network (CNN)]")
    print("  Mathematical Foundation:")
    print("    • Convolutional layers: small filters (3×3) slide across image")
    print("    • Pooling layers: reduce spatial dimensions (summarize regions)")
    print("    • Fully connected layers: classify based on learned features")
    print("    • Loss: binary cross-entropy for abnormality presence")
    print("  Why CNN?")
    print("    - Spatial locality: nearby pixels correlated (tumors are local)")
    print("    - Parameter sharing: same filter detects abnormality at any location")
    print("    - Translation invariance: tumor in any part of image detected")
    print("    - Deep: learns hierarchical features (edges → shapes → organs → tumors)")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 10,000 images with expert annotations")
    print("  • Validation: 2,000 images reviewed by multiple radiologists")
    print("  • Challenging: Need rare abnormalities (~5-10% prevalence)")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: ROC-AUC (threshold-independent performance)")
    print("    - Trade-off: sensitivity (catch cancers) vs specificity (avoid false alarms)")
    print("  Secondary:")
    print("    - Sensitivity at 95% specificity: catch 95% of abnormalities")
    print("    - Per-abnormality type performance (cancers harder than fractures)")
    print("    - Comparison to radiologist performance (human baseline)")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Regulatory: FDA clearance required ($1-5M, 2-3 years)")
    print("  2. Liability: Missed diagnosis → lawsuit (system must be very safe)")
    print("  3. Generalization: Model trained on one hospital's images, may fail elsewhere")
    print("  4. Practice change: Radiologists resist automation (job security)")
    print("  5. Explainability: 'Why did network flag this?'' (deep learning is 'black box')")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Radiologist shortage: high demand, workforce strained")
    print("  • Productivity: AI systems detect abnormalities 10% faster")
    print("  • Hospital scale: 50 radiologists saving 1 hour/day = 50 FTE saved")
    print("  • Cost: 50 FTE × $300k/year = $15M savings/year")
    print("  • Model cost: $3-5M development, $500k/year operations")
    print("  • ROI: 3:1 to 5:1 over 3 years")


def print_healthcare_problem_5():
    """Patient Mortality Risk Scoring"""
    print("\n" + "="*70)
    print("HEALTHCARE PROBLEM #5: ICU Mortality Risk Prediction")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("ICU patients have high mortality risk. Predicting who will likely die")
    print("lets doctors allocate resources effectively and discuss prognosis with")
    print("families, informing end-of-life care decisions.")
    
    print("\n[FEATURES]")
    print("  • Vital signs (heart rate, blood pressure, temperature, respiratory rate)")
    print("  • Laboratory values (glucose, creatinine, WBC, platelets, pH)")
    print("  • SOFA score (Sequential Organ Failure Assessment, 0-24)")
    print("  • APACHE score (Acute Physiology and Chronic Health Eval, 0-71)")
    print("  • Glasgow Coma Scale (eye/verbal/motor, 3-15)")
    print("  • Primary diagnosis (categorical: sepsis, cardiogenic shock, etc.)")
    print("  • Mechanical ventilation required (binary)")
    print("  • Vasopressor support (binary)")
    print("  • Renal replacement therapy (binary)")
    print("  • Age (continuous)")
    print("  • Chronic conditions (comorbidities count)")
    print("  • Prior ICU admission (binary, frequency)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Binary: In-hospital mortality (1) or discharge alive (0)")
    print("  Timeframe: mortality within 90 days of ICU admission")
    
    print("\n[ALGORITHM: Elastic Net Logistic Regression]")
    print("  Mathematical Foundation:")
    print("    • Logistic regression: P(death) = 1 / (1 + e^(-z))")
    print("    • Elastic Net: combines L1 (Lasso) and L2 (Ridge) penalties")
    print("    • Sparsity: L1 component zeros out weak predictors")
    print("    • Stability: L2 component keeps correlated features")
    print("  Why Elastic Net?")
    print("    - Interpretable: clinicians understand linear risk factors")
    print("    - Parsimonious: few features = easy to implement in ICU")
    print("    - Stable: handles correlated vital signs well")
    print("    - Fast: real-time predictions (milliseconds)")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 50,000 ICU admissions")
    print("  • Multi-hospital: data from 5+ hospitals (generalization)")
    print("  • Mortality rate: ~10-15% in general ICU population")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Calibration (predicted probability = actual mortality rate)")
    print("    - If we predict 70% mortality, ~70% of such patients should die")
    print("    - Calibration error: max difference between prediction and actual")
    print("  Secondary:")
    print("    - ROC-AUC: discrimination (separate high-risk from low-risk)")
    print("    - Net Benefit: clinical utility at different probability thresholds")
    print("    - Brier Score: mean squared prediction error")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Ethical: Should algorithm influence treatment withdrawal decisions?")
    print("  2. Self-fulfilling: If predicted to die, fewer interventions → worse outcome")
    print("  3. Privacy: Hospitals reluctant to share patient data")
    print("  4. Disparity: Model must be fair across racial/ethnic/socioeconomic groups")
    print("  5. Validation: Requires 90-day follow-up (slow feedback loops)")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • ICU cost: $10,000+ per day, average 5-day stay = $50k per patient")
    print("  • Unnecessary care: 10% of ICU patients pursue aggressive care knowing outcome futile")
    print("  • Healthcare savings: Better prognosis discussions, less wasted resource")
    print("  • Value: 5% reduction in futile ICU care = $100M+ per large health system")
    print("  • Model cost: $500k development, $100k/year maintenance")


# ============================================================================
# RETAIL/SHOPPING PROBLEMS
# ============================================================================

def print_retail_problem_1():
    """Customer Churn Prediction"""
    print("\n" + "="*70)
    print("RETAIL PROBLEM #1: Customer Churn Prediction")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Retail companies lose customers constantly. Instead of losing customers")
    print("silently, predict who will leave. Target retention campaigns at at-risk")
    print("customers before they switch to competitors.")
    
    print("\n[FEATURES]")
    print("  • Customer tenure (months with company, 0-120)")
    print("  • Monthly spending (continuous, $0-$1000)")
    print("  • Purchase frequency (count per month, 0-50)")
    print("  • Product categories purchased (binary features per category)")
    print("  • Customer service contacts (count, 0-20)")
    print("  • Complaint rate (number of complaints, 0-10)")
    print("  • Return rate (% of purchases returned, 0-100)")
    print("  • Last purchase recency (days since last order, 0-365)")
    print("  • Email engagement (open rate, click rate, 0-100)")
    print("  • Website session count (logins per month, 0-50)")
    print("  • Mobile app usage (sessions per month, 0-50)")
    print("  • Loyalty program member (binary)")
    print("  • Online reviews submitted (count, 0-50)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Binary: Customer will churn (1) or retain (0) in next 6 months")
    
    print("\n[ALGORITHM: Gradient Boosting (XGBoost)]")
    print("  Mathematical Foundation:")
    print("    • Sequential trees: each tree corrects errors of prior trees")
    print("    • Gradient descent: optimize with shrinkage parameter α (learning rate)")
    print("    • Regularization: prevent overfitting via tree depth and leaf weight")
    print("    • Feature importance: identifies key churn drivers")
    print("  Why XGBoost?")
    print("    - Handles class imbalance (churn ~5%, not 50%)")
    print("    - Feature importance: shows key churn drivers for marketing")
    print("    - Non-linear interactions: automatically captures cross-feature effects")
    print("    - Fast inference: milliseconds per prediction")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 100,000 customer records (12+ months each)")
    print("  • Churn proportion: ~5-10% (realistic for retail)")
    print("  • Data freshness: must reflect current customer behavior")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Precision at 80% recall")
    print("    - Of customers predicted to churn, how many actually churn?")
    print("    - Goal: >50% precision (don't waste retention budget)")
    print("  Secondary:")
    print("    - Lift: How much better than random targeting (goal: 3-5x)")
    print("    - Profit curve: $value per correct prediction - cost per campaign")
    print("    - Time-to-churn: days until churn (helps timing intervention)")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Feedback loops: Targeting changes behavior (selection bias)")
    print("  2. Causal inference: Does offer cause customer to stay? (RCT needed)")
    print("  3. Campaign effectiveness: Hard to isolate model's contribution")
    print("  4. Privacy: Customers may object to churn targeting")
    print("  5. Fairness: Model shouldn't discriminate (lower retention rules for some groups)")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Lifetime value: Customer stays 5 years @ $30/month = $1800 total")
    print("  • Acquisition cost: $50 to acquire new customer")
    print("  • Churn: 20% annual rate = 20,000 lost customers/year for 100k customer base")
    print("  • Loss: 20,000 × $1800 = $36M annual churn cost")
    print("  • Model target: 3% churn reduction (6,000 saved customers)")
    print("  • Value: 6,000 × $1800 = $10.8M save → ROI 50:1 (extremely valuable)")


def print_retail_problem_2():
    """Product Recommendation Engine"""
    print("\n" + "="*70)
    print("RETAIL PROBLEM #2: Personalized Product Recommendation System")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Online stores overwhelm customers with product choices. Personalized")
    print("recommendations increase conversion (purchases) and average order value.")
    print("Customers also enjoy shopping easier when shown relevant products.")
    
    print("\n[FEATURES]")
    print("  • Customer purchase history (vector of past products)")
    print("  • Product viewed (browsing history)")
    print("  • Similar customers' purchases (collaborative filtering signal)")
    print("  • Product features (category, brand, price, rarity)")
    print("  • Product ratings (average & count of reviews)")
    print("  • Seasonal trends (products trending now)")
    print("  • Customer demographics (age, location, inferred from behavior)")
    print("  • Time since purchase (product age, stock age)")
    print("  • Click-through data (which products clicked vs ignored)")
    print("  • Cart abandonment (products left in cart but not purchased)")
    print("  • Search queries (what customer is looking for)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Ranked list of top-10 recommended products with predicted scores")
    print("  Alternative: predicted click-through rate (CTR) per product")
    
    print("\n[ALGORITHM: Hybrid Recommendation System]")
    print("  Mathematical Foundation:")
    print("    • Collaborative filtering: user-item similarity matrix")
    print("    • Content-based: product feature similarity (cosine)")
    print("    • Hybrid: weighted combination of two approaches")
    print("    • Context: time-of-day, season, trending adjustments")
    print("  Why Hybrid?")
    print("    - Collaborative captures taste preferences (user taste)")
    print("    - Content-based handles new products (cold-start)")
    print("    - Hybrid leverages both signals for better coverage")
    print("    - Contextual features capture temporal patterns")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 1M customer-product interactions over 6 months")
    print("  • Customer-product matrix: very sparse (0.1% filled)")
    print("  • Active users: 100k+ users with 10+ purchases each")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Click-Through Rate (CTR)")
    print("    - % of recommended products that customer clicks on")
    print("    - Real-world engagement metric")
    print("  Secondary:")
    print("    - Conversion rate (click→purchase), goal >1%")
    print("    - Average order value (higher price items recommended?)")
    print("    - Diversity (not same category over and over)")
    print("    - Novelty (recommend products customer wouldn't find themselves)")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Cold-start: New users have no history (use item popularity)")
    print("  2. Data sparsity: Most customer-product pairs unknown")
    print("  3. Feedback loops: Popular items get more recommendations → more popular")
    print("  4. Diversity: Optimize for both accuracy and product diversity")
    print("  5. Fairness: Shouldn't favor expensive/high-margin products unfairly")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Baseline: 2% of browsing customers make purchase")
    print("  • With recommendations: 3% conversion rate (50% improvement)")
    print("  • Scale: 1M browsers/month, 10k additional purchases/month")
    print("  • Average order value: $40")
    print("  • Revenue increase: 10k × $40 = $400k/month")
    print("  • Annual: $4.8M new revenue for ~$200k development + $50k/year ops")


def print_retail_problem_3():
    """Dynamic Pricing Optimization"""
    print("\n" + "="*70)
    print("RETAIL PROBLEM #3: Dynamic Pricing Algorithm")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Fixed prices leave money on the table. Demand varies by season, competitor")
    print("pricing, and inventory. Dynamic pricing adjusts prices in real-time to")
    print("maximize revenue while maintaining competitiveness and profit margins.")
    
    print("\n[FEATURES]")
    print("  • Product demand (recent sales velocity, historical demand pattern)")
    print("  • Inventory level (units in stock, storage cost)")
    print("  • Competitor prices (real-time price monitoring, 0-20 competitors)")
    print("  • Product age (days since introduction, stage in lifecycle)")
    print("  • Seasonality (month/quarter, holiday flags)")
    print("  • Customer segment (price-sensitive vs. premium customers)")
    print("  • Price elasticity (historical % change in demand per 1% price change)")
    print("  • Product category (luxury/necessity affects price sensitivity)")
    print("  • Customer acquisition cost (profit margin needed to justify acquisition)")
    print("  • Product rarity (exclusive items command premium)")
    print("  • Expiration date (perishables need clearance before spoilage)")
    print("  • Reviews/quality (higher-rated products sustain higher prices)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Decision: Recommended selling price per product")
    print("  Goal: maximize profit = final_price * expected_quantity_sold")
    
    print("\n[ALGORITHM: Reinforcement Learning (Contextual Bandits)]")
    print("  Mathematical Foundation:")
    print("    • Multi-armed bandit: explore pricing, exploit good prices")
    print("    • Context: product/market features determine exploration strategy")
    print("    • Reward: profit per sale (price × probability of sale)")
    print("    • Epsilon-greedy: 80% exploit best known price, 20% explore new prices")
    print("  Why RL/Bandits?")
    print("    - Real-time learning: prices adjust daily based on outcomes")
    print("    - Context-aware: different strategies per product category")
    print("    - Exploration-exploitation: balance learning new prices vs using best prices")
    print("    - Online: learns continuously without retraining batch models")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 10,000 product-price pairs with sales outcomes")
    print("  • Historical: Price changes and corresponding demand (6-12 months)")
    print("  • Real-time: continuous feed of competitor prices and own sales")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Revenue per unit (actual price × quantity sold)")
    print("    - Compare to baseline (fixed pricing)")
    print("  Secondary:")
    print("    - Profit margin (revenue - cost)")
    print("    - Inventory turnover (time to sell products)")
    print("    - Price competitiveness (position vs competitors)")
    print("    - Customer satisfaction (volume discounts for loyal customers)")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Price wars: Competitors adjust prices (feedback loops)")
    print("  2. Customer perception: Too many price changes → loses trust")
    print("  3. Fairness: Same product different prices for similar customers (controversial)")
    print("  4. Regulation: Some jurisdictions prohibit dynamic pricing")
    print("  5. Demand model: Hard to estimate price elasticity accurately")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Baseline: fixed prices, 1M products × $30 average = $30M revenue")
    print("  • Elasticity: demand decreases ~2% per 1% price increase (typical)")
    print("  • Optimization: average price increase 5%, demand decrease 10%")
    print("  • New revenue: (1M * 0.9) * $31.50 = $28.35M (net loss if badly designed!)")
    print("  • Better strategy: Fine-grained optimization per segment")
    print("  • Realistic gain: 2-5% revenue increase = $600k-1.5M on $30M base")
    print("  • Complex model: more modest gains but reduces downside risk")


def print_retail_problem_4():
    """Market Basket Analysis"""
    print("\n" + "="*70)
    print("RETAIL PROBLEM #4: Market Basket Analysis (Association Rules)")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Customers who buy Product A often buy Product B. By understanding these")
    print("patterns, stores can (1) improve product placement, (2) bundle products")
    print("for discounts, (3) recommend complementary products, and (4) make")
    print("targeted promotions that increase basket size.")
    
    print("\n[FEATURES]")
    print("  • Transaction history (transactions × products, binary/count matrix)")
    print("  • Product features:")
    print("    - Product ID")
    print("    - Category (cereal, produce, dairy, etc.)")
    print("    - Aisle location")
    print("    - Price")
    print("    - Brand")
    print("  • Customer features:")
    print("    - Customer ID")
    print("    - Shopping frequency")
    print("    - Loyalty program member")
    print("    - Store location")
    
    print("\n[TARGET/OUTPUT]")
    print("  Association rules: if customer buys {Product A, Product B}")
    print("  then likely also buys {Product C}")
    print("  Format: {milk, bread} → {butter} (confidence: 65%, support: 5%)")
    
    print("\n[ALGORITHM: Apriori Association Rule Mining]")
    print("  Mathematical Foundation:")
    print("    • Support: P(A ∩ B) = fraction of transactions with both")
    print("    • Confidence: P(B|A) = P(A ∩ B) / P(A) = if A then B with what prob?")
    print("    • Lift: confidence / P(B) = strength of association (>1 means positive)")
    print("    • Apriori property: if {A,B,C} frequent, then all subsets also frequent")
    print("  Why Apriori?")
    print("    - Interpretable: rules are human-readable (bread→butter)")
    print("    - Exhaustive: finds ALL strong associations (not sampling)")
    print("    - Proven: decades of use in retail (foundational algorithm)")
    print("    - Actionable: directly informs store layout and promotions")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 100,000 transactions")
    print("  • Grocery store: 1000-2000 transactions/day, 50k/month = 6M/year")
    print("  • Products: 5,000-30,000 distinct items in supermarket")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Lift (association strength)")
    print("    - Lift > 1: products positively associated (buy together)")
    print("    - Lift = 1: independent (no association)")
    print("    - Lift < 1: negative association (substitute products)")
    print("  Secondary:")
    print("    - Confidence: if rule applied, accuracy of prediction")
    print("    - Support: prevalence of rule (rule applies to X% of customers)")
    print("    - Conviction: strength of implication (A implies B)")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Data sparsity: millions of products but most never bought together")
    print("  2. Spurious correlations: may detect random associations")
    print("  3. Actionability: not all rules lead to profitable actions")
    print("  4. Seasonality: rules change (Christmas cookies in winter, not summer)")
    print("  5. Feedback loops: promoting bundles changes buying patterns")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Basket size lift: 5-10% increase in average transaction value")
    print("  • Supermarket sales: $1M/week, 5% increase = $50k/week = $2.6M/year")
    print("  • Model cost: $50k development (simpler algorithm)")
    print("  • ROI: 50:1 (extremely profitable, simple implementation)")


def print_retail_problem_5():
    """Product Demand Forecasting"""
    print("\n" + "="*70)
    print("RETAIL PROBLEM #5: Product Demand Forecasting")
    print("="*70)
    
    print("\n[PROBLEM DEFINITION]")
    print("Retailers must stock products in advance. Too much inventory →")
    print("storage costs and markdowns. Too little → stock-outs and lost sales.")
    print("Forecasting demand helps optimize inventory levels to balance these")
    print("competing costs and maximize profit.")
    
    print("\n[FEATURES]")
    print("  • Historical sales (daily/weekly sales for past 2-3 years)")
    print("  • Seasonality (month, holiday periods, day-of-week)")
    print("  • Trend (growing/declining product category)")
    print("  • Price (current price, price of competitors)")
    print("  • Promotions (discount %, advertisement spend, duration)")
    print("  • Weather (temperature, precipitation, affects certain products)")
    print("  • External events (sports events, holidays, news)")
    print("  • Product features (new release, popular brand, quality tier)")
    print("  • Store characteristics (location, size, customer demographics)")
    print("  • Competitor actions (competitor sales, stockouts, promotions)")
    
    print("\n[TARGET/OUTPUT]")
    print("  Forecast: predicted sales quantity for next week/month")
    print("  Ideally: probabilistic forecast (mean ± confidence interval)")
    print("  Format: next_week_demand = 1000 ± 100 units (90% confidence)")
    
    print("\n[ALGORITHM: ARIMA / Exponential Smoothing + External Regressors]")
    print("  Mathematical Foundation:")
    print("    • ARIMA: Auto-Regressive Integrated Moving Average")
    print("    •   AR (p): today's value depends on past p values")
    print("    •   I (d): differencing to remove trends")
    print("    •   MA (q): error terms of past q steps")
    print("    • ARIMA formula: y_t = φ₁y_{t-1} + θ₁e_{t-1} + promotion + γ*price + noise")
    print("    • Alternative: Facebook Prophet (handles seasonality, holidays)")
    print("  Why ARIMA + external features?")
    print("    - Captures temporal dependencies (yesterday predicts today)")
    print("    - Incorporates external factors (price, promotions)")
    print("    - Interpretable: understand how each feature affects demand")
    print("    - Proven: decades of use in supply chain forecasting")
    
    print("\n[DATASET SIZE]")
    print("  • Minimum: 2-3 years of daily/weekly sales (700+ data points per product)")
    print("  • Products: forecast separately per SKU (0-3000 products in store)")
    print("  • External data: prices, promotions, weather, events (aligned to sales)")
    
    print("\n[EVALUATION METRICS]")
    print("  Primary: Mean Absolute Percentage Error (MAPE)")
    print("    - MAPE = (1/n) * Σ|actual - predicted| / |actual| * 100")
    print("    - Goal: MAPE < 10-15% for grocery, higher for volatile products")
    print("  Secondary:")
    print("    - MAE: Mean Absolute Error (units, not percentage)")
    print("    - Pinball loss: penalizes under/over-forecasting differently")
    print("    - Profit: cost of stockout vs overstock in $ terms")
    
    print("\n[DEPLOYMENT CHALLENGES]")
    print("  1. Data quality: missing data, stockouts (unobserved demand)")
    print("  2. Structural breaks: new promotions, competitors enter/exit")
    print("  3. Intermittent demand: slow-moving items, many zero-sales weeks")
    print("  4. Multi-step ahead: harder to forecast 4 weeks ahead vs 1 week")
    print("  5. Feedback loops: forecast affects inventory → affects actual sales")
    
    print("\n[FINANCIAL IMPACT]")
    print("  • Overstock cost: 20% markup lost to clearance = $10/unit")
    print("  • Stockout cost: lost profit + customer dissatisfaction = $20/unit")
    print("  • Forecast error: 20% MAPE = some units overstocked, some stockout")
    print("  • Optimization: reduce MAPE to 10% = balance costs better")
    print("  • Savings: 100,000 units/year, 10% improvement = $150k/year")
    print("  • Model cost: $100k development + $20k/year operations")
    print("  • ROI: 1.5:1 year 1, much better in years 2+")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Header
    print("\n" + "="*70)
    print("ML IDEA GENERATOR: 15 REAL-WORLD PROBLEMS ACROSS 3 DOMAINS")
    print("="*70)
    
    # Print all problems
    print_college_problem_1()
    print_college_problem_2()
    print_college_problem_3()
    print_college_problem_4()
    print_college_problem_5()
    
    print_healthcare_problem_1()
    print_healthcare_problem_2()
    print_healthcare_problem_3()
    print_healthcare_problem_4()
    print_healthcare_problem_5()
    
    print_retail_problem_1()
    print_retail_problem_2()
    print_retail_problem_3()
    print_retail_problem_4()
    print_retail_problem_5()
    
    # Footer
    print("\n" + "="*70)
    print("ALL 15 PROBLEMS DISPLAYED")
    print("Each problem covers: definition, features, output, algorithm,")
    print("dataset size, metrics, deployment challenges, and financial impact.")
    print("="*70)
