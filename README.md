Medical Data Visualizer

This project is part of the Data Analysis with Python certification from freeCodeCamp. It involves visualizing and making calculations from medical examination data using pandas, matplotlib, and seaborn.

Project Description

In this project, we explore the relationship between cardiac disease, body measurements, blood markers, and lifestyle choices using a dataset collected from medical examinations. The dataset includes information such as age, height, weight, blood pressure, cholesterol levels, glucose levels, and lifestyle choices like smoking, alcohol intake, and physical activity.

Dataset

The dataset used in this project is medical_examination.csv, which contains the following columns:

Age: age (in days)
Height: height (in cm)
Weight: weight (in kg)
Gender: categorical code (1 for male, 2 for female)
Systolic blood pressure: ap_hi (int)
Diastolic blood pressure: ap_lo (int)
Cholesterol: cholesterol (1: normal, 2: above normal, 3: well above normal)
Glucose: gluc (1: normal, 2: above normal, 3: well above normal)
Smoking: smoke (binary)
Alcohol intake: alco (binary)
Physical activity: active (binary)
Presence or absence of cardiovascular disease: cardio (binary)
Tasks

The following tasks were completed in this project:

Add an Overweight Column:

Calculated the BMI (weight divided by height squared) and determined if a person is overweight (BMI > 25).
Added a column overweight with values 0 (not overweight) and 1 (overweight).
Normalize Data:

Normalized the cholesterol and gluc columns so that 0 is always good and 1 is always bad.
Create a Categorical Plot:

Converted the data into long format.
Created a chart using seaborn's catplot() to show counts of good and bad outcomes for cholesterol, glucose, alcohol intake, physical activity, and smoking, split by the presence or absence of cardiovascular disease.
Clean the Data:

Filtered out incorrect data where diastolic pressure is higher than systolic pressure.
Removed outliers for height and weight based on the 2.5th and 97.5th percentiles.
Draw a Heat Map:

Calculated a correlation matrix.
Plotted the correlation matrix using seaborn's heatmap() with a mask for the upper triangle.
File Structure

medical_data_visualizer.py: Contains the code for data processing and visualization.
test_module.py: Contains unit tests for the project.
medical_examination.csv: The dataset used in the project.

How to Run

Clone the repository:

git clone https://github.com/yourusername/medical-data-visualizer.git
cd medical-data-visualizer

Install the required libraries:

pip install pandas matplotlib seaborn

Run the project:

python medical_data_visualizer.py

Conclusion

This project provides a comprehensive analysis of medical examination data, demonstrating the ability to preprocess data, perform statistical analysis, and create visualizations using Python libraries such as pandas, matplotlib, and seaborn.
