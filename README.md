# Yuva Intern – Week 1 Data Science Project 📊

## 📌 Project Overview

This project is part of my **Yuva Intern Data Science – Week 1** task.

The project focuses on **data preprocessing, data cleaning and exploratory data analysis** using the **Adult Census Income Dataset**.

The dataset was processed using Python and Pandas to handle missing values, detect outliers, remove duplicates, encode categorical variables and create visualizations.

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn
- UCI Machine Learning Repository

---

## 🔍 Data Preprocessing Steps

### 1. Dataset Loading
Loaded the Adult Census Income Dataset using Python.

### 2. Dataset Inspection
Checked:

- Number of rows and columns
- Column names
- Data types
- Dataset structure

### 3. Missing Value Handling
Identified missing values and handled missing entries represented by `?`.

Categorical missing values were replaced using the **mode** of the respective column.

### 4. Outlier Detection
Used the **Interquartile Range (IQR)** method to identify potential outliers in numerical columns.

### 5. Duplicate Removal
Removed duplicate records from the dataset.

### 6. Categorical Encoding
Converted categorical variables into numerical format using **One-Hot Encoding**.

### 7. Data Visualization
Created visualizations to understand the dataset, including:

- Age Distribution
- Age vs Hours per Week
- Capital Gain Distribution

### 8. Clean Dataset
The processed dataset was exported as:

`cleaned_adult_dataset.csv`

---

## 📂 Project Files

```text
Yuva_Intern_Week1/
│
├── README.md
│
├── week1_adult_preprocessing.py
│
└── dataset/
