# 🩸 Blood Donation Prediction  
*Logistic regression analysis of donor behavior and retention likelihood*  

[![Medium](https://img.shields.io/badge/Read_on-Medium-black?logo=medium)](https://medium.com/@danliontario/predicting-blood-donation-likelihood-with-logistic-regression-4a93f2d85028)  
[![Portfolio](https://img.shields.io/badge/View-Portfolio-blue?logo=react)](https://dlport.web.app/)  

---

## 📖 Project Overview  
Blood donation plays a crucial role in healthcare, yet maintaining repeat donors remains a persistent challenge.  
This project analyzes data from the **UCI Blood Transfusion Service Center dataset** to predict whether a donor will give blood again based on their donation history.

The goal was to identify key behavioral predictors — such as **recency**, **frequency**, and **total volume donated** — and to build a **logistic regression model** that estimates donation likelihood.

---

## 🧪 Methods  
- **Tools**: Python (Pandas, Scikit-learn, Seaborn, Matplotlib), MySQL Workbench  
- **Model**: Logistic Regression  
- **Metrics**: Accuracy (74%), ROC AUC (0.76)  
- **Dataset**: 748 donor records with five key variables (recency, frequency, monetary, months, donate)

The full analysis and discussion are available on [Medium](https://medium.com/@danliontario/predicting-blood-donation-likelihood-with-logistic-regression-4a93f2d85028).  

---

## 📊 Key Findings  
- **Recency** was the strongest predictor — recent donors were far more likely to donate again.  
- **Frequency** and **monetary total** were perfectly correlated.  
- Model achieved **74% accuracy** and **ROC AUC = 0.76**.  
- Class imbalance (76% non-donors vs 24% donors) limited recall for positive cases.

---

