from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from Huggingface_CLIP_main import *
from Huggingface_CLIP_utils import *
from Evaluate_CLIP import *
from transformers import *
from tqdm import tqdm
import evaluate
from datasets import load_dataset
from transformers import pipeline
from transformers import ViTFeatureExtractor, ViTForImageClassification
import urllib.parse as parse
import os

# set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI()

# Load your trained CLIP model here
# %%
# the encoder model that process the image and return the image features
# encoder_model = "WinKawaks/vit-small-patch16-224"
# encoder_model = "google/vit-base-patch16-224"
# encoder_model = "google/vit-base-patch16-224-in21k"
encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
# the decoder model that process the image features and generate the caption text
# decoder_model = "bert-base-uncased"
# decoder_model = "prajjwal1/bert-tiny"
decoder_model = "gpt2"
# load the model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, decoder_model
).to(device)

best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)

# Create a cache for storing previous inference results
cache = {}

@app.post("/find_similar")
async def find_similar_image(file: UploadFile):
    image = Image.open(file.file)
    # Process the image and find the most similar image and text using the CLIP model
    similar_image, similar_text = find_most_similar_image_and_text(best_model, image)
    
    # Store the result in the cache
    cache[file.filename] = (similar_image, similar_text)
    
    return {"similar_image": similar_image, "similar_text": similar_text}

@app.get("/cached_result/{filename}")
async def get_cached_result(filename: str):
    if filename in cache:
        return {"similar_image": cache[filename][0], "similar_text": cache[filename][1]}
    else:
        return {"error": "Result not found in the cache"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
