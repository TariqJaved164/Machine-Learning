# Student Exam Score Forecasting

An end-to-end Machine Learning project designed to predict a student's final exam performance using behavioral metrics, daily habits, and past academic records. By leveraging a tree-based ensemble method, this system serves as an early-warning tool for educators to identify students who may need academic intervention before formal testing begins.

## 📌 Project Overview

In traditional educational settings, academic risks are often discovered only after low exam scores are recorded. This project builds a predictive analytics pipeline that transforms student habit tracking into proactive insights. 

Using a dataset of 10,000 student records, the system evaluates six behavioral and historical indicators to forecast a continuous final score out of 100. The underlying model is optimized using a **Gradient Boosting Regressor**, achieving an **$R^2$ score of 0.72** and a **Mean Absolute Error (MAE) of ~5.96 marks**.

---

## ✨ Key Features

* **Continuous Score Forecasting:** Predicts the exact continuous score of a student out of 100 rather than broad letter grades.
* **Feature Importance Extraction:** Identifies and ranks the exact impact of habits (e.g., study hours, attendance, sleep) on final grades to help educators give data-driven advice.
* **Early Intervention Design:** Outlines a data framework that allows teachers to run predictive simulations during the middle of an academic term.
* **Production-Ready Serialization:** Serializes the final trained model into a compact pipeline pickle file (`.pkl`) for rapid API deployment or frontend dashboard integration.

---

## 📊 Dataset Structure

The model utilizes the following attributes to calculate predictions:

| Feature Name | Data Type | Description |
|---|---|---|
| `study_hours` | Integer | Average hours spent studying per day |
| `attendance` | Integer | Percentage of classes attended (40% - 100%) |
| `sleep_hours` | Integer | Average hours of sleep per night |
| `internet_usage` | Integer | Estimated daily internet hours |
| `assignments_completed`| Integer | Total assignments submitted out of 20 |
| `previous_score` | Integer | Average score achieved in previous terms |
| **`exam_score` (Target)**| Float | Final exam outcome out of 100 |

---

