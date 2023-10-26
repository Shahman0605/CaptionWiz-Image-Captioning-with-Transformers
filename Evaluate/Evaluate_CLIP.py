"""
### MODEL EVALUATION

"""
import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from Huggingface_CLIP_utils import *
from torch.optim import AdamW
import evaluate
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import pipeline
from transformers import ViTFeatureExtractor, ViTForImageClassification
import urllib.parse as parse
import os

# set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def get_evaluation_metrics(model, dataset):
  model.eval()
  #
  dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)

  n_test_steps = len(dataloader)
  
  predictions, labels = [], []
  
  test_loss = 0.0
  for batch in tqdm(dataloader, "Evaluating"):
      
      pixel_values = batch["pixel_values"]
      label_ids = batch["labels"]
      
      outputs = model(pixel_values=pixel_values, labels=label_ids)
      # outputs = model.generate(pixel_values=pixel_values, max_length=max_length)
      
      loss = outputs.loss
      test_loss += loss.item()
      
      logits = outputs.logits.detach().cpu()
      
      predictions.extend(logits.argmax(dim=-1).tolist())
      
      labels.extend(label_ids.tolist())

  eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
  # compute the metrics
  metrics = compute_metrics(eval_prediction)
  # add the test_loss to the metrics
  metrics["test_loss"] = test_loss / n_test_steps
  return metrics
  
best_checkpoint = 3000
best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)

metrics = get_evaluation_metrics(best_model, test_dataset)
print(metrics)

finetuned_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
finetuned_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
finetuned_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

get_evaluation_metrics(finetuned_model, test_dataset)

"""
## Compare with other model
"""

image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning")
image_captioner.model = image_captioner.model.to(device)

get_evaluation_metrics(image_captioner.model, test_dataset)