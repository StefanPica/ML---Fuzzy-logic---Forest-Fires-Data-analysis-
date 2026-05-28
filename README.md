# Forest Fire Analysis: ML, Fuzzy Logic & Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00.svg)

## 📌 Overview
This repository contains a comprehensive study investigating whether advanced analytical methods can extract meaningful structure from a real-world forest-fire dataset. The analysis is built on three methodological pillars:
1. **Predictive Machine Learning** (Regression vs. Classification).
2. **Fuzzy Inference Systems** (Modeling environmental fire spread potential).
3. **Unsupervised Outlier Detection**.

## 📊 Dataset
The analysis uses the **Montesinho Forest Fires dataset**, focusing on core meteorological variables including Temperature, Relative Humidity (RH), Wind Speed, Rainfall, and the Initial Spread Index (ISI). ->

https://www.kaggle.com/datasets/anitarostami/montesinho-forest-fire-prediction-dataset

## 🧠 Methodology & Experimental Design
All experiments were implemented in Google Colab using a standardized and fully reproducible computational pipeline.
* **Regression Framework:** Attempted to predict the continuous burned area using Random Forest, XGBoost, and MLP Regressors.
* **Classification Framework:** Reframed the problem from fire-size prediction to environmental-severity classification (LOW vs. MEDIUM spread conditions).
* **Fuzzy Logic:** Developed a rule-based inference system based on meteorological favorability to cross-validate data-driven models.
* **Explainability:** Applied **SHAP** (SHapley Additive exPlanations) to interpret model feature importance.

## 📈 Key Findings
1. **Regression is Infeasible:** Predicting exact burned area from meteorological data failed (yielding near-zero or negative $R^2$ values). The near-zero correlation between the fuzzy *FireSpreadPotential* and burned area confirms that environmental favorability does not translate directly to observed fire extent.
2. **Classification is Highly Successful:** By reframing the target to classify environmental severity (LOW vs. MEDIUM), the models achieved exceptional accuracy:
   * **XGBoost:** 99%
   * **Random Forest:** 98%
   * **MLP:** 96%
3. **Model & Rule-Based Alignment:** The ML classification models align closely with the Fuzzy Logic system, demonstrating strong consistency between data-driven and rule-based approaches.
4. **Feature Importance:** SHAP analysis identified Temperature, Humidity, Wind, and ISI as the dominant factors influencing spread-favorable conditions.
5. **Structural Discoveries via Outlier Detection:** Unsupervised outlier detection revealed a structural explanation for the modeling results. LOW-severity fires are environmentally irregular (containing a disproportionate number of outliers), whereas MEDIUM-severity fires occupy a coherent, predictable meteorological space. 

## 🚀 How to Run the Experiment
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost scikit-fuzzy simpful shap
   
