<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeoRisk – Neonatal Health Risk Assessment</title>

<style>
body {
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  line-height: 1.7;
}

header {
  background: linear-gradient(135deg, #1e3a8a, #2563eb);
  color: white;
  padding: 70px 20px;
  text-align: center;
}

header h1 {
  font-size: 2.6rem;
  margin-bottom: 10px;
}

header p {
  max-width: 900px;
  margin: auto;
  font-size: 1.1rem;
  opacity: 0.95;
}

.badges img {
  margin: 15px 5px 0;
}

.container {
  max-width: 1100px;
  margin: auto;
  padding: 40px 20px;
}

section {
  margin-bottom: 70px;
}

h2 {
  font-size: 2rem;
  color: #1e293b;
  margin-bottom: 15px;
}

.card {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

pre {
  background: #020617;
  color: #e5e7eb;
  padding: 20px;
  border-radius: 10px;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

table th, table td {
  padding: 12px;
  border-bottom: 1px solid #e5e7eb;
  text-align: left;
}

table th {
  background: #f1f5f9;
}

ul {
  margin-left: 20px;
}

.button {
  display: inline-block;
  margin: 10px 10px 0 0;
  padding: 12px 22px;
  background: #2563eb;
  color: white;
  text-decoration: none;
  border-radius: 999px;
  font-weight: 600;
  transition: 0.2s;
}

.button:hover {
  background: #1e40af;
}

footer {
  background: #020617;
  color: #cbd5f5;
  padding: 40px 20px;
  text-align: center;
}

footer span {
  color: #22c55e;
  font-weight: bold;
}
</style>
</head>

<body>

<header>
  <h1>Predictive Analytics for Neonatal Health Risk Assessment</h1>
  <p>A Machine Learning Approach to Early Detection of Critical Health Conditions in Newborns</p>

  <div class="badges">
    <a href="https://doi.org/10.5281/zenodo.18383330">
      <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18383330-blue?style=for-the-badge&logo=zenodo">
    </a>
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python">
  </div>

  <p style="margin-top:25px;">
    <strong>NeoRisk</strong> – An open-source, reproducible machine learning pipeline that predicts daily neonatal health risk
    (Healthy or At Risk) using longitudinal monitoring data from newborns.
  </p>

  <a class="button" href="https://github.com/yourusername/NeoRisk">View on GitHub</a>
  <a class="button" href="https://doi.org/10.5281/zenodo.18383330">Zenodo DOI</a>
</header>

<div class="container">

<section>
  <h2>✨ Project Highlights</h2>
  <div class="card">
    <ul>
      <li>Daily risk prediction using vitals, growth, feeding, and jaundice</li>
      <li>Data leakage detection (ROC AUC ≈ 1.0 → realistic 0.85–0.94)</li>
      <li>Temporal modeling using LSTM (7-day sliding windows)</li>
      <li>Class imbalance handling with SMOTE</li>
      <li>Published and cited with Zenodo DOI</li>
    </ul>
  </div>
</section>

<section>
  <h2>📄 Abstract</h2>
  <div class="card">
    <p>
      NeoRisk predicts neonatal health risk on a daily basis using longitudinal data from 100 newborns.
      Initial tabular models showed near-perfect performance due to leakage.
      After correction, realistic performance ranged between 0.85–0.94 ROC AUC.
    </p>
    <p>
      A leakage-free LSTM model using 7-day historical sequences captured physiological dynamics,
      making NeoRisk clinically meaningful and deployment-ready.
    </p>
  </div>
</section>

<section>
  <h2>🚀 Quick Start</h2>
  <div class="card">
<pre>
git clone https://github.com/yourusername/NeoRisk.git
cd NeoRisk
pip install -r requirements.txt
jupyter notebook NewBornNeoRisk.ipynb
</pre>
  </div>
</section>

<section>
  <h2>🤖 Models & Performance</h2>
  <div class="card">
    <table>
      <tr>
        <th>Model</th><th>ROC AUC</th><th>Accuracy</th><th>Recall</th><th>Notes</th>
      </tr>
      <tr><td>Random Forest + SMOTE</td><td>~1.000</td><td>0.9983</td><td>1.000</td><td>Leakage</td></tr>
      <tr><td>XGBoost + SMOTE</td><td>~1.000</td><td>0.9983</td><td>1.000</td><td>Leakage</td></tr>
      <tr><td>Logistic Regression</td><td>0.9569</td><td>0.8967</td><td>0.8625</td><td>Baseline</td></tr>
      <tr><td>XGBoost (No Leakage)</td><td>0.85–0.94</td><td>Realistic</td><td>High</td><td>Clinically usable</td></tr>
      <tr><td>LSTM (7-day)</td><td>0.87–0.93</td><td>High</td><td>High</td><td>Temporal</td></tr>
    </table>
  </div>
</section>

<section>
  <h2>📌 Citation</h2>
  <div class="card">
<pre>
@misc{abbas2026neorisk,
  author = {Agha Wafa Abbas},
  title  = {Predictive Analytics for Neonatal Health Risk Assessment},
  year   = {2026},
  doi    = {10.5281/zenodo.18383330}
}
</pre>
  </div>
</section>

</div>

<footer>
  Built with <span>❤️</span> for neonatal health and reproducible science<br>
  © January 2026 — Agha Wafa Abbas
</footer>

</body>
</html>
