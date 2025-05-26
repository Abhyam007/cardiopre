# cardiopre
# CardioRisk: AI-Powered Cardiovascular Disease Prediction App

**CardioRisk** is an intelligent health assistant that uses machine learning to assess cardiovascular disease risk based on patient metrics. Built with **Streamlit** for frontend interactivity and **scikit-learn** for predictive modeling, it also offers an API integration via **IBM Watsonx.ai**.

---

## 🚀 Overview

CardioRisk provides a fast and intuitive interface for predicting heart disease risk using medical input values. It supports:

- **Live web-based risk prediction** via a Streamlit dashboard  
- **Trained ML model** using Random Forest  
- **Input features** commonly used in cardiovascular diagnostics  
- **Deployment-ready model API** via IBM Watsonx.ai  

---

## 🧠 Key Features

### 🩺 Interactive Risk Prediction  
Users can input health metrics like age, blood pressure, cholesterol, and lifestyle factors to get immediate risk predictions.

### 🧠 Machine Learning Based  
Uses a trained `RandomForestClassifier` to predict presence/absence of cardiovascular disease.

### 🌐 IBM Watsonx.ai API Integration  
The same model is deployed to the cloud with a **public REST API endpoint** for external applications.

### 📊 Preprocessing Included  
Data is pre-cleaned using Z-score filtering, standard scaling, and one-hot encoding.

---

## 🛠️ Technology Stack

| Layer        | Technology                    |
|--------------|-------------------------------|
| Frontend     | Streamlit                     |
| Model        | Random Forest (scikit-learn)  |
| Deployment   | IBM Watsonx.ai + Streamlit App |
| Data Prep    | Pandas, Scipy, StandardScaler |
| Storage      | joblib for saving model/scaler |

---

## 🧩 Architecture

```plaintext
┌────────────┐     Web Input     ┌─────────────────────┐
│  Streamlit │ ─────────────────►│   appcar.py UI      │
└────────────┘                   └────────┬────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │  ML Model (RF Classifier)│
                            └────────────┬─────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │  Prediction + Risk Score │
                            └──────────────────────────┘
🧪 Installation & Usage
1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/cardiorisk.git
cd cardiorisk
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or manually install:

bash
Copy
Edit
pip install streamlit pandas scikit-learn joblib
3️⃣ Run the application
Use this command to avoid Streamlit path issues:

bash
Copy
Edit
python -m streamlit run appcar.py
App will open in your browser at http://localhost:8501

🌐 IBM Watsonx.ai API Access
This app is also deployed as a REST API using IBM Watson ML.

🔗 Public Endpoint
bash
Copy
Edit
POST https://us-south.ml.cloud.ibm.com/ml/v4/deployments/9ba8b1ff-2318-45a7-81ce-e1c89a3f9c4f/predictions?version=2021-05-01
📤 Example Request (JSON)
json
Copy
Edit
{
  "input_data": [
    {
      "fields": ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"],
      "values": [[44, 1, 153, 93, 140, 90, 1, 1, 0, 0, 1]]
    }
  ]
}
📥 Example Response
json
Copy
Edit
{
  "predictions": [
    {
      "fields": ["prediction", "probability"],
      "values": [[1, [0.28, 0.72]]]
    }
  ]
}
📦 File Structure
bash
Copy
Edit
cardiorisk/
│
├── appcar.py             # Streamlit frontend + prediction logic
├── rf_model.pkl          # Trained Random Forest model
├── scaler.pkl            # StandardScaler for preprocessing
├── up1.csv               # Optional: Training dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project info and usage
🛡️ Disclaimer
This tool is intended for educational and research purposes only. It is not a substitute for professional medical advice or diagnosis.

🙌 Credits
ML Training: scikit-learn

Frontend: Streamlit

Cloud Deployment: IBM Watsonx.ai

