# Predictive Analytics for Neonatal Health Risk Assessment: A Machine Learning Approach to Early Detection of Critical Health Conditions in Newborns
A Machine Learning Approach to Early Detection of Critical Health Conditions in Newborns

<p align="center">
  <img src="https://via.placeholder.com/1200x400/1e3a8a/ffffff?text=NeoRisk+-+Saving+Newborn+Lives+with+AI" alt="NeoRisk Banner" width="100%">
  <br><br>
  <a href="https://doi.org/10.5281/zenodo.18383330">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18383330-blue?style=for-the-badge&logo=zenodo" alt="DOI">
  </a>
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--learn-orange?style=for-the-badge" alt="Frameworks">
  <img src="https://img.shields.io/badge/Healthcare%20AI-Open%20Source-success?style=for-the-badge" alt="Open Source">
</p>

<p align="center">
  <strong>NeoRisk</strong> – An open-source, reproducible machine learning pipeline that predicts daily neonatal health risk (Healthy or At Risk) using longitudinal monitoring data from newborns.
</p>

<hr>

## ✨ Project Highlights

- **Daily Risk Prediction** using vital signs, growth metrics, feeding patterns, and jaundice levels  
- **Detects Data Leakage** – Initial ROC AUC ≈ 1.0 → Realistic 0.85–0.94 after correction  
- **Temporal Modeling** with LSTM on 7-day sliding windows – captures physiological dynamics  
- **Class Imbalance Handling** with SMOTE – ensures reliable detection of rare “At Risk” cases  
- **Published & Cited** – Zenodo DOI: [10.5281/zenodo.18383330](https://doi.org/10.5281/zenodo.18383330)  
- **Ideal for** ML engineers, healthcare AI researchers, and academicians building impactful portfolios

<hr>

## 📄 Abstract

Infant morbidity and mortality in the first month of life is a significant contributor to global infant deaths. This work presents **NeoRisk**, a machine learning model that predicts neonatal risk level (Healthy or At Risk) on a daily basis using longitudinal monitoring data from 100 newborns.

Initial tabular models (Logistic Regression, Random Forest, XGBoost) showed near-perfect results (ROC AUC ≈ 1.0). However, after systematically removing leaking features (especially jaundice values), realistic performance was found in the **0.85–0.94 ROC AUC** range. A time-series LSTM model using 7-day historical sequences was developed to capture linked physiological dynamics, avoiding leakage and addressing class imbalance.

Key predictors (post-leakage): weight change, gestational age.  
**NeoRisk** provides a clinically relevant, reproducible risk stratification pipeline and demonstrates the power of longitudinal deep learning in neonatal care.

**Keywords**: Neonatal risk prediction · Machine learning · Data leakage · Longitudinal data · Time-series modeling · LSTM · Jaundice · Newborn monitoring · Predictive analytics · Class imbalance

<hr>

## 👨‍🏫 Author & Affiliations

**Agha Wafa Abbas**  
Lecturer, School of Computing  
- University of Portsmouth, Winston Churchill Ave, Southsea, Portsmouth PO1 2UP, United Kingdom  
- Arden University, Coventry, United Kingdom  
- Pearson, London, United Kingdom  
- IVY College of Management Sciences, Lahore, Pakistan  

📧 Emails:  
- agha.wafa@port.ac.uk  
- awabbas@arden.ac.uk  
- wafa.abbas.lhr@rootsivy.edu.pk  

<p align="center">
  <a href="https://www.linkedin.com/in/agha-wafa-abbas">LinkedIn</a> • 
  <a href="https://github.com/yourusername">GitHub</a> • 
  <a href="https://doi.org/10.5281/zenodo.18383330">Zenodo DOI</a>
</p>

<hr>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/NeoRisk.git
cd NeoRisk

# Install dependencies
pip install -r requirements.txt

# Launch the complete notebook
jupyter notebook NewBornNeoRisk.ipynb

# 🛠️ What's Inside
File / Folder,Description
NewBornNeoRisk.ipynb,Complete ML pipeline: EDA → Preprocessing → Tabular + LSTM models → Evaluation
newborn_health_monitoring_with_risk.csv,"Longitudinal dataset (3000 records, 100 newborns, 30 days)"
neorisk_xgboost_no_leak.pkl,Best leakage-corrected XGBoost model (ready for inference)
neorisk_final_comparison.csv,Final model performance table
Predictive Analytics...Newborns.pdf,Full research paper (26 pages)
requirements.txt,All dependencies for reproducibility



# Launch the complete notebook
jupyter notebook NewBornNeoRisk.ipynb

# Models and Performance

Model,ROC AUC,Accuracy,Recall (At Risk),Notes
Random Forest + SMOTE,~1.000,0.9983,1.000,Near-perfect – but strong leakage
XGBoost + SMOTE,~1.000,0.9983,1.000,Near-perfect – but strong leakage
Logistic Regression,0.9569,0.8967,0.8625,Strong linear baseline
XGBoost (No Jaundice Leakage),0.85–0.94,Realistic,High,Clinically usable after leakage removal
LSTM (7-day sequences),0.87–0.93,High,High,Temporal modeling – no leakage features

# Visualizations, confusion matrices, ROC curves, and feature importance plots are generated live in the notebook.

🔍 Key Contributions for ML Engineers & Academicians

Systematic data leakage diagnosis and correction
Realistic benchmarking of tabular vs. time-series deep learning in healthcare
Practical handling of class imbalance in rare-event medical prediction
Fully reproducible pipeline with saved models and Zenodo DOI
Production-ready inference example for real-time neonatal monitoring

# Citation 
@misc{abbas2026neorisk,
  author       = {Agha Wafa Abbas},
  title        = {Predictive Analytics for Neonatal Health Risk Assessment: A Machine Learning Approach to Early Detection of Critical Health Conditions in Newborns},
  year         = {2026},
  doi          = {10.5281/zenodo.18383330},
  url          = {https://doi.org/10.5281/zenodo.18383330},
  howpublished = {GitHub repository},
}
💙 Thank You
Thank you for exploring NeoRisk.
This project is built with the hope of saving newborn lives through responsible, transparent, and clinically meaningful AI.
If you find this work useful, please ⭐ the repository and cite the DOI.
Questions, collaborations, or improvements?
Feel free to open an issue or email me directly.
Built with ❤️ for neonatal health and reproducible science.

  Footer Banner

  Last updated: January 2026
