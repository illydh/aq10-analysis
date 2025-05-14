import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
from model_utils import predict_asd

# Load OpenAI key from environment
oai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = oai_key

app = FastAPI(title="ASD Clinical Support API")

class PatientData(BaseModel):
    age: int
    gender: int       # 0=F,1=M
    A1_Score: int
    A2_Score: int
    A3_Score: int
    A4_Score: int
    A5_Score: int
    A6_Score: int
    A7_Score: int
    A8_Score: int
    A9_Score: int
    A10_Score: int

@app.post("/predict")
async def predict(data: PatientData):
    # Assemble feature vector in order
    feat = [
        data.A1_Score, data.A2_Score, data.A3_Score, data.A4_Score,
        data.A5_Score, data.A6_Score, data.A7_Score, data.A8_Score,
        data.A9_Score, data.A10_Score, data.age, data.gender
    ]
    res = predict_asd(feat)

    # Create GPT summary
    prompt = (
        f"The ASD model predicts a probability of {res['probability']:.2f} "
        f"(decision: {'Positive' if res['decision'] else 'Negative'}). "
        "Provide a concise summary for a doctor and suggestions for next steps."
    )
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    summary = completion.choices[0].message.content

    return {**res, "summary": summary}