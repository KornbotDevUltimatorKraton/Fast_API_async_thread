from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import asyncio

# Load the SBERT model (can be done at the start of the application)
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

# Pydantic model for request body
class TextRequest(BaseModel):
    text: str

# Route to process text input and return embeddings
@app.post("/encode/")
async def encode_text(request: TextRequest):
    # Processing the request asynchronously
    embeddings = await asyncio.to_thread(model.encode, request.text)
    return {"embeddings": embeddings.tolist()}

# To run the FastAPI app:
# uvicorn app_name:app --reload
