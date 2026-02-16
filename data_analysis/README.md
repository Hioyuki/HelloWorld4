Data Analysis Project: Personality Traits & Tech Salary Trends
📝 Project Description
This project explores two distinct datasets to uncover hidden patterns in human behavior and the global tech job market. Using Python and Pandas, I analyzed how personality types influence communication and identified the key technical skills that drive elite salaries in the tech industry.

❓ Data Analysis Questions
Personality Analysis: Do specific personality types (such as INTJ) demonstrate unique communication patterns in terms of vocabulary and post length?

Salary Analysis: What are the true drivers of elite salaries ($1M+ USD), and how do factors like education, remote work, and specific technical skills impact these figures?

🛠 Technologies & Libraries
Python 3.x

Pandas: For data manipulation, filtering, and aggregation.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization.

Scikit-learn: For TF-IDF vectorization (Data Conversion).

📊 Demonstrated Requirements
In this software, I have implemented the following data analysis techniques:

Filtering: Narrowed down salary data to USD currency to ensure a fair comparison.

Aggregation: Calculated Mean and Median salaries, as well as job satisfaction scores.

Sorting: Ranked personality types by average post length and identified top keywords.

Data Conversion: Converted raw text posts into numerical TF-IDF values.

Visualization: Created charts including a Log-scale Box Plot to visualize salary distribution and outliers.

💡 Results & Insights
Skill-Driven Success: Elite earners (making over $1M) frequently list Docker, Kubernetes, and GCP as primary skills.

Education vs. Salary: The median salary for 'Self-taught' individuals is nearly identical to those with a 'PhD', indicating that practical skills are highly valued.

Work-Life Balance: Interestingly, elite earners and average earners report almost the same work hours and job satisfaction.

🏃 How to Run the Software
Ensure you have the datasets (mbti_clean_train.csv and tech_jobs_salaries.csv) in the data/ folder.

Install dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn
Run the analysis:

Bash
python scripts/final_analysis.py
🎬 Video Demo
Please click the link below to watch the code walkthrough and software demo:

👉 Watch the Video Demo on YouTube