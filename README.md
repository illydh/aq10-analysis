# Autism Diagnosis Prediction using Transformer Models

## 📌 Overview

This project develops a transformer-based model to predict Autism Spectrum Disorder (ASD) diagnosis using AQ-10 questionnaire responses and demographic data. The system analyzes patterns in behavioral traits to assist clinicians in early ASD detection across different age groups.

**Key Features**:
- **Transformer Inference**  
  Loads your best PyTorch Lightning checkpoint and infers ASD risk from AQ10 + demographic data.
- **GPT-Powered Summaries**  
  Uses the OpenAI API to generate a human-readable summary of the model’s prediction and actionable next steps.
- **Clean, Modular Structure**  
  - **Machine Learning** code (data prep, model, training, evaluation) lives alongside  
  - **Backend** (FastAPI) and  
  - **Frontend** (React)  
- **Easy Local Development**  
  Launch backend and frontend with simple commands.

## 📊 Dataset

The [Autism Screening Adult Dataset](https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults) contains:
- 10 AQ-10 questionnaire items (binary responses)
- Demographic features (age, gender)
- Clinical diagnosis labels
- 704 samples

**Sample Data**:

| Age | Gender | A1_Score | ... | A10_Score | Result | Diagnosis |
|-----|--------|----------|-----|-----------|--------|-----------|
| 25  | 1      | 1        | ... | 0         | 7      | 1         |

## 🛠️ Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/autism-diagnosis-prediction.git
cd autism-diagnosis-prediction
```

2. Create and activate virtual environment:
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies: 
```
pip install -r requirements.txt
```

## 🚀 Usage

### Training the Model
```
python src/train.py
```

### Evaluation
```
python src/evaluate.py
```

### Install & configure backend
```
cd backend
pip install -r requirements.txt
# set your environment variables:
export OPENAI_API_KEY="sk-..."
export MODEL_CHECKPOINT_PATH="app/best.ckpt"

# to activate, run the following line in a separate process
uvicorn app.main:app --reload --port 8000
```

### Install & configure frontend
```
cd ../frontend
npm install
# create a .env.local if desired:
# REACT_APP_API_URL=http://localhost:8000

# to activate, run the following line in a separate process
npm start
```


## 🧠 Model Architecture
**Sample Data**:
1. **Feature Embedding Layer**: Projects questionnaire items into 64D space
2. **Transformer Encoder**: 4 layers with 4 attention heads
3. **Dual Prediction Heads**:
    - Diagnosis prediction (clinical ASD)
    - Classification prediction (ASD traits)

# 📦 API Reference

- **Request JSON**:
```
{
  "age": 8,
  "gender": "M",
  "responses": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
}
```
