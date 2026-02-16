import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# =================================================================
# FUNCTION 1: MBTI Personality & Text Analysis
# =================================================================

def analyze_mbti_personality(file_path):
    """
    Analyzes MBTI dataset to identify language patterns and post lengths.
    Demonstrates: Filtering, Aggregation (Mean), and Data Conversion (TF-IDF).
    """
    print("\n" + "="*50)
    print("STEP 1: MBTI PERSONALITY TEXT ANALYSIS")
    print("="*50)

    try:
        # Load data with specific encoding for text posts
        df = pd.read_csv(file_path, encoding='latin1')
        print(f"✅ Successfully loaded MBTI data. Total records: {len(df)}")

        # Question 1: Does post length vary by personality type?
        df['post_length'] = df['posts'].fillna('').apply(len)
        avg_length = df.groupby('type')['post_length'].mean().sort_values(ascending=False)
        
        print("\n[Average Post Length by Type]")
        print(avg_length.head())

        # Question 2: What specific words define the INTJ type?
        # Data Conversion: Convert text to numerical TF-IDF values
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(df['posts'].fillna(''))
        words = vectorizer.get_feature_names_out()

        # Identify indices for INTJ
        int_indices = df[df['type'] == 'INTJ'].index
        # Calculate relative usage compared to average
        overall_mean = np.asarray(X.mean(axis=0)).ravel()
        intj_mean = np.asarray(X[int_indices].mean(axis=0)).ravel()
        
        # Avoid division by zero
        relative_importance = np.divide(intj_mean, overall_mean, out=np.zeros_like(intj_mean), where=overall_mean!=0)
        
        # Sort and get Top 10
        top_indices = relative_importance.argsort()[-10:][::-1]
        print("\n[Top 10 Signature Words for INTJ (Relative to Others)]")
        for i in top_indices:
            print(f"{words[i]}: {relative_importance[i]:.2f}")

        # Visualization
        plt.figure(figsize=(10, 5))
        sns.barplot(x=avg_length.index, y=avg_length.values, palette='viridis')
        plt.title('Average Post Length per MBTI Type')
        plt.xticks(rotation=45)
        plt.show()

    except Exception as e:
        print(f"❌ Error in MBTI Analysis: {e}")

# =================================================================
# FUNCTION 2: Tech Job Salary Analysis
# =================================================================

def analyze_tech_salaries(file_path):
    """
    Analyzes Tech Salaries to identify high-earner traits.
    Demonstrates: Filtering (USD), Sorting, and Aggregation (Median vs Mean).
    """
    print("\n" + "="*50)
    print("STEP 2: TECH JOB SALARY ANALYSIS")
    print("="*50)

    try:
        df = pd.read_csv(file_path)
        # Filtering for USD to normalize data
        df_usd = df[df['currency'] == 'USD'].copy()
        print(f"✅ Successfully loaded Salary data. Analyzing {len(df_usd)} USD records.")

        # Question 1: How does education level impact salary?
        edu_order = ['Self-taught', 'Diploma', 'Bachelor', 'Master', 'PhD']
        edu_salary = df_usd.groupby('education_level')['salary_local_currency'].median().reindex(edu_order)
        print("\n[Median Salary by Education Level (USD)]")
        print(edu_salary)

        # Question 2: Who are the 'Elite' earners ($1M+)?
        rich_group = df_usd[df_usd['salary_local_currency'] >= 1000000].copy()
        print(f"\n📊 Elite Earners ($1M+): {len(rich_group)} people ({(len(rich_group)/len(df_usd))*100:.2f}%)")

        print("\n[Top 5 Job Titles for Elite Earners]")
        print(rich_group['job_title'].value_counts().head(5))

        print("\n[Top 5 Skills for Elite Earners]")
        print(rich_group['primary_skill'].value_counts().head(5))

        # Comparison of Work-Life Balance
        print("\n[Elite Group vs Normal Group: Work-Life Balance]")
        is_rich = df_usd['salary_local_currency'] >= 1000000
        comparison = df_usd.groupby(is_rich)[['work_hours_per_week', 'job_satisfaction_score']].mean()
        comparison.index = ['Normal (<$1M)', 'Elite ($1M+)']
        print(comparison)

        # Visualization: Boxplot for Education Levels
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_usd, x='education_level', y='salary_local_currency', order=edu_order, palette='Set3')
        plt.yscale('log') # Log scale helps visualize outliers and median together
        plt.title('Salary Distribution by Education (Log Scale)')
        plt.show()

    except Exception as e:
        print(f"❌ Error in Salary Analysis: {e}")

# =================================================================
# MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    # Path to your CSV files
    mbti_file = 'mbti_clean_train.csv'
    salary_file = 'tech_jobs_salaries.csv'

    # Run Analysis Functions
    analyze_mbti_personality(mbti_file)
    analyze_tech_salaries(salary_file)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)