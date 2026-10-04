from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

MODEL_DIR = "./model"
LABELS = ["negative", "neutral", "positive"]
app=FastAPI()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

class TicketRequest(BaseModel):
    text: str
@app.post("/predict")
def predict(req:TicketRequest):
    inputs = tokenizer(req.text, return_tensors="tf", truncation=True, max_length=128)
    output=model(**inputs)
    logits=output.logits
    probs=tf.nn.softmax(logits,axis=-1)
    pred_index = tf.argmax(probs, axis=-1)
    confidence = tf.reduce_max(probs, axis=-1)
    pred_index = int(pred_index.numpy()[0])
    confidence = float(confidence.numpy()[0])
    label = LABELS[pred_index]

    return {"label": label, "confidence": confidence}
@app.get("/health")
def health():
    return {"status": "ok"}