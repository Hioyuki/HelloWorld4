# Data Analysis Project  
## Personality Traits and Tech Salary Trends

---

## Project Overview

This project explores two distinct datasets to uncover hidden patterns in human communication behavior and global tech salary trends.

Using Python and modern data science libraries, I analyzed how personality traits influence communication patterns and identified the key drivers behind elite tech salaries (over $1M USD).

---

## Research Questions

### 1. Personality Analysis

Do specific personality types (such as INTJ) demonstrate unique communication patterns in:

- Vocabulary usage
- Post length
- Word frequency trends

### 2. Salary Analysis

What drives elite salaries (over $1M USD)?

- Does education level significantly impact salary?
- Does remote work influence compensation?
- Which technical skills are most common among elite earners?

---

## Technologies and Libraries

- Python 3.x
- Pandas (data manipulation and aggregation)
- NumPy (numerical operations)
- Matplotlib (data visualization)
- Seaborn (statistical visualization)
- Scikit-learn (TF-IDF vectorization)

---

## Data Analysis Techniques Implemented

### Filtering
Filtered the salary dataset to USD currency to ensure fair comparison.

### Aggregation
Calculated mean and median salaries.  
Compared job satisfaction across income groups.

### Sorting
Ranked personality types by average post length.  
Identified top keywords per personality type.

### Data Conversion
Converted raw text posts into numerical TF-IDF vectors.

### Visualization
Created multiple charts including a log-scale box plot to visualize salary distribution and outliers.

---

## Results and Insights

### Skill-Driven Success

Elite earners (over $1M USD) frequently list the following skills:

- Docker  
- Kubernetes  
- GCP  

These skills appear significantly more often in the elite salary group.

---

### Education vs Salary

The median salary of self-taught individuals is nearly identical to those holding a PhD.

This suggests that practical skills and applied knowledge are valued more than formal academic credentials in high-income tech roles.

---

### Work-Life Balance

Elite earners and average earners report nearly identical:

- Weekly work hours
- Job satisfaction levels

Higher salary does not necessarily correlate with longer working hours or lower satisfaction.

---

## How to Run the Project

### 1. Place Datasets in the Following Directory

data/
├── mbti_clean_train.csv
└── tech_jobs_salaries.csv


### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

3. Run the Analysis
python scripts/final_analysis.py


## Video Demonstration

Watch the full code walkthrough and software demonstration on YouTube:

[Watch the Video Demo on YouTube](https://your-youtube-link-here)

