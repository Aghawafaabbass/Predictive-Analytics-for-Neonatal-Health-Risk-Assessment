# NeoRisk: Predictive Analytics for Neonatal Health Risk Assessment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18383330.svg)](https://doi.org/10.5281/zenodo.18383330)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A machine learning framework for early detection of critical health conditions in newborns through longitudinal monitoring data analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Paper](#research-paper)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🔬 Overview

NeoRisk is a comprehensive machine learning framework designed to predict neonatal health risk levels (Healthy or At Risk) on a daily basis using longitudinal monitoring data from the first 30 days of life. The project addresses critical challenges in neonatal predictive analytics, including:

- **Data leakage detection and mitigation**
- **Class imbalance correction**
- **Temporal dynamics modeling**
- **Clinical interpretability**

### Why NeoRisk?

Neonatal mortality accounts for approximately 47% of all under-five deaths globally (~2.4 million deaths in 2023). Early detection of deteriorating health conditions can significantly improve outcomes, but traditional clinical scoring systems often fail to capture the dynamic nature of newborn physiology.

## ✨ Key Features

- **Systematic Leakage Investigation**: Identifies and removes features that artificially inflate model performance
- **Multi-Model Comparison**: Evaluates tabular models (Logistic Regression, Random Forest, XGBoost) against time-series models (LSTM)
- **Class Imbalance Handling**: Implements SMOTE (Synthetic Minority Over-sampling Technique) specifically on training data
- **Temporal Modeling**: LSTM-based architecture using 7-day sliding windows for sequential prediction
- **Clinical Relevance**: Prioritizes high recall for At-Risk cases to minimize missed deteriorations
- **Full Reproducibility**: Fixed random seeds, documented preprocessing, and saved model artifacts

## 📄 Research Paper

**Title:** Predictive Analytics for Neonatal Health Risk Assessment: A Machine Learning Approach to Early Detection of Critical Health Conditions in Newborns

**Author:** Agha Wafa Abbas  
**Affiliations:**
- School of Computing, University of Portsmouth, UK
- School of Computing, Arden University, UK
- School of Computing, Pearson, London, UK
- School of Computing, IVY College of Management Sciences, Lahore, Pakistan

**DOI:** [10.5281/zenodo.18383330](https://doi.org/10.5281/zenodo.18383330)

**Abstract:** This paper presents NeoRisk, a machine learning model to predict neonatal risk levels using longitudinal monitoring data. Initial results with tabular models showed near-perfect performance (ROC AUC ≈ 1.000), but systematic leakage removal revealed realistic performance in the 0.85-0.94 range. The study demonstrates the critical importance of data leakage detection and the benefits of longitudinal deep learning strategies in neonatal care.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU for LSTM training acceleration

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neorisk.git
cd neorisk
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Dependencies

Core libraries:
- `pandas` >= 1.3.0
- `numpy` >= 1.21.0
- `scikit-learn` >= 1.0.0
- `imbalanced-learn` >= 0.8.0
- `xgboost` >= 1.5.0
- `torch` >= 1.10.0
- `matplotlib` >= 3.4.0
- `seaborn` >= 0.11.0

See `requirements.txt` for complete list with specific versions.

## 📊 Dataset

### Overview

The dataset consists of simulated neonatal health records designed to replicate real-world NICU monitoring:

- **Total Records:** 3,000 (100 newborns × 30 days)
- **Features:** 25 variables (static birth data + dynamic daily measurements)
- **Target:** Binary classification (Healthy: 86.73%, At Risk: 13.27%)

### Features

**Static Variables:**
- Gestational age (weeks)
- Birth weight (kg)
- Birth length (cm)
- Birth head circumference (cm)
- Apgar score
- Gender

**Dynamic Variables (Daily):**
- Weight, length, head circumference
- Vital signs (temperature, heart rate, respiratory rate, oxygen saturation)
- Feeding type and frequency
- Urine output and stool count
- Jaundice level (mg/dL)
- Immunization status
- Reflex assessment

### Data Availability

The synthetic dataset used in this study is included in the repository. For privacy and ethical considerations, no real patient data is included.

## 🏗️ Model Architecture

### Tabular Models

1. **Logistic Regression**
   - Linear baseline model
   - Class weight balancing
   - Interpretable coefficients

2. **Random Forest**
   - 200 estimators
   - Max depth: 10
   - Feature importance via impurity decrease

3. **XGBoost** (Best performing)
   - 200 boosting rounds
   - Max depth: 6
   - Learning rate: 0.1
   - Early stopping on validation set

### Time-Series Model

**LSTM Architecture:**
- Input: 3D tensor (batch_size, 7 days, 21 features)
- 2 LSTM layers (hidden size: 64, dropout: 0.2)
- Fully connected output layer with sigmoid activation
- Loss: BCEWithLogitsLoss (class-weighted)
- Optimizer: Adam (lr: 0.001)

## 📈 Results

### Leakage-Corrected Performance

After systematic removal of leaking features (primarily jaundice level):

| Model | ROC AUC | Accuracy | Precision (At Risk) | Recall (At Risk) | F1 Score |
|-------|---------|----------|---------------------|------------------|----------|
| **XGBoost** | **0.947** | **0.938** | **0.781** | **0.900** | **0.836** |
| Random Forest | 0.938 | 0.925 | 0.742 | 0.875 | 0.803 |
| Logistic Regression | 0.912 | 0.892 | 0.612 | 0.825 | 0.703 |
| LSTM (7-day) | 0.921 | 0.915 | 0.714 | 0.850 | 0.776 |

### Key Findings

1. **Data Leakage Impact:** Initial models showed near-perfect performance (AUC ≈ 1.000) due to jaundice level feature leakage
2. **Realistic Performance:** Post-leakage correction yields AUC in 0.85-0.94 range, consistent with literature
3. **Top Predictors (Post-Leakage):**
   - Weight change from birth
   - Heart rate (bpm)
   - Respiratory rate (bpm)
   - Gestational age (weeks)
4. **High Sensitivity:** All models maintain recall >0.825 for At-Risk class (critical for clinical use)
5. **Temporal Value:** LSTM provides comparable performance while capturing sequential dynamics

## 💻 Usage

### Running the Complete Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook NewBornNeoRisk.ipynb
```

The notebook contains:
1. Data loading and exploration
2. Preprocessing pipeline
3. Leakage investigation
4. Model training (tabular + LSTM)
5. Evaluation and visualization
6. Feature importance analysis

### Quick Start Example

```python
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('models/neorisk_xgboost_no_leak.pkl')

# Load and preprocess new data
data = pd.read_csv('your_neonatal_data.csv')
# ... apply preprocessing pipeline ...

# Make predictions
risk_probabilities = model.predict_proba(data)[:, 1]
risk_labels = model.predict(data)

# Classify newborns
high_risk_cases = data[risk_labels == 1]
print(f"Identified {len(high_risk_cases)} high-risk cases")
```

### Training From Scratch

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load preprocessed data
X_train, X_test, y_train, y_test = load_preprocessed_data()

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test)
# ... calculate metrics ...
```

## 📁 Project Structure

```
neorisk/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   └── NewBornNeoRisk.ipynb          # Main analysis notebook
│
├── data/
│   ├── neorisk_final_comparison.csv  # Model comparison results
│   └── neorisk_model_comparison.csv  # Detailed metrics
│
├── models/
│   └── neorisk_xgboost_no_leak.pkl   # Trained XGBoost model
│
├── src/
│   ├── preprocessing.py               # Data preprocessing functions
│   ├── models.py                      # Model architectures
│   ├── evaluation.py                  # Evaluation metrics
│   └── utils.py                       # Utility functions
│
├── results/
│   ├── figures/                       # Generated plots
│   └── metrics/                       # Performance metrics
│
└── docs/
    └── paper.pdf                      # Research paper
```

## 🔄 Reproducibility

This project prioritizes reproducibility through:

1. **Fixed Random Seeds:** All stochastic processes use fixed seeds
   - train_test_split: `random_state=42`
   - SMOTE: `random_state=42`
   - Model initialization: `random_state=42`
   - PyTorch: `torch.manual_seed(42)`

2. **Environment Management:**
   - `requirements.txt` with pinned versions
   - Python 3.8+ compatibility

3. **Documentation:**
   - Detailed markdown comments in notebook
   - Step-by-step preprocessing pipeline
   - Clear model hyperparameters (Table 3 in paper)

4. **Model Artifacts:**
   - Saved trained models via joblib
   - Preprocessing scalers included

5. **Computational Requirements:**
   - Tabular models: <30 seconds on CPU
   - LSTM: 2-3 minutes with GPU acceleration
   - Platform tested: Google Colab

### Running Exact Replications

```bash
# Clone repository
git clone https://github.com/yourusername/neorisk.git
cd neorisk

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/NewBornNeoRisk.ipynb
```

## 📚 Citation

If you use NeoRisk in your research, please cite:

```bibtex
@article{abbas2025neorisk,
  title={Predictive Analytics for Neonatal Health Risk Assessment: A Machine Learning Approach to Early Detection of Critical Health Conditions in Newborns},
  author={Abbas, Agha Wafa},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.18383330},
  url={https://doi.org/10.5281/zenodo.18383330}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- External validation on real NICU datasets
- Additional model architectures (Transformers, Attention mechanisms)
- Multi-modal data integration (waveforms, imaging)
- Fairness audits across demographic subgroups
- Clinical deployment tools
- Documentation improvements

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is provided for research and educational purposes only. It is NOT intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📧 Contact

**Agha Wafa Abbas**

- Email: agha.wafa@port.ac.uk
- Email: awabbas@arden.ac.uk
- Email: wafa.abbas.lhr@rootsivy.edu.pk

**Affiliations:**
- University of Portsmouth, UK
- Arden University, UK
- Pearson, London, UK
- IVY College of Management Sciences, Lahore, Pakistan

## 🙏 Acknowledgments

- Dataset inspired by real-world NICU monitoring protocols
- Thanks to the open-source community for scikit-learn, XGBoost, PyTorch, and related libraries
- Special thanks to reviewers and contributors

## 📊 Project Status

- ✅ Research paper published
- ✅ Code fully documented
- ✅ Models trained and evaluated
- 🔄 Seeking external validation datasets
- 🔄 Planning clinical deployment study

---

**Keywords:** Neonatal risk prediction, Machine learning, Data leakage, Longitudinal data, Time-series modeling, LSTM, Newborn monitoring, Predictive analytics, Class imbalance

**Last Updated:** January 2025
