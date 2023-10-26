import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from Huggingface_CLIP_utils import *
import Models
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

# load the model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, decoder_model
).to(device)


# tokenizer = AutoTokenizer.from_pretrained(decoder_model)
tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
# tokenizer = BertTokenizerFast.from_pretrained(decoder_model)

image_processor = ViTImageProcessor.from_pretrained(encoder_model)

if "gpt2" in decoder_model:
  tokenizer.pad_token = tokenizer.eos_token # pad_token_id as eos_token_id
  model.config.eos_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  model.config.decoder_start_token_id = tokenizer.cls_token_id
  model.config.pad_token_id = tokenizer.pad_token_id

max_length = 32 
coco_dataset_ratio = 50 
train_ds = load_dataset("HuggingFaceM4/COCO", split=f"train[:{coco_dataset_ratio}%]")
valid_ds = load_dataset("HuggingFaceM4/COCO", split=f"validation[:{coco_dataset_ratio}%]")
test_ds = load_dataset("HuggingFaceM4/COCO", split="test")
len(train_ds), len(valid_ds), len(test_ds)

# remove the images with less than 3 dimensions (possibly grayscale images)
train_ds = train_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)
valid_ds = valid_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)
test_ds = test_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)


train_dataset = train_ds.with_transform(preprocess)
valid_dataset = valid_ds.with_transform(preprocess)
test_dataset  = test_ds.with_transform(preprocess)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

num_epochs = 2 
batch_size = 64 

for item in train_dataset:
  print(item["labels"].shape)
  print(item["pixel_values"].shape)
  break

# define our data loaders
train_dataset_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
test_dataset_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)


optimizer = AdamW(model.parameters(), lr=1e-5)


%load_ext tensorboard
%tensorboard --logdir ./image-captioning/tensorboard

summary_writer = SummaryWriter(log_dir="./image-captioning/tensorboard")
n_train_steps = num_epochs * len(train_dataset_loader)
n_valid_steps = len(valid_dataset_loader)
current_step = 0
save_steps = 1000

for epoch in range(num_epochs):
    # set the model to training mode
    model.train()
    # initialize the training loss
    train_loss = 0
    for batch in tqdm(train_dataset_loader, "Training", total=len(train_dataset_loader), leave=False):
      if current_step % save_steps == 0:
        print(f"\nValidation at step {current_step}...\n")
        model.eval()
        predictions, labels = [], []
        valid_loss = 0
        for batch in valid_dataset_loader:
            pixel_values = batch["pixel_values"]
            label_ids = batch["labels"]
            outputs = model(pixel_values=pixel_values, labels=label_ids)
            loss = outputs.loss
            valid_loss += loss.item()
            logits = outputs.logits.detach().cpu()
            predictions.extend(logits.argmax(dim=-1).tolist())
            labels.extend(label_ids.tolist())
        
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        
        metrics = compute_metrics(eval_prediction)
        # print the stats
        print(f"\nEpoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f}, " + 
              f"Valid Loss: {valid_loss / n_valid_steps:.4f}, BLEU: {metrics['bleu']:.4f},  " + 
              f"ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}\n")
        # log the metrics
        summary_writer.add_scalar("valid_loss", valid_loss / n_valid_steps, global_step=current_step)
        summary_writer.add_scalar("bleu", metrics["bleu"], global_step=current_step)
        summary_writer.add_scalar("rouge1", metrics["rouge1"], global_step=current_step)
        summary_writer.add_scalar("rouge2", metrics["rouge2"], global_step=current_step)
        summary_writer.add_scalar("rougeL", metrics["rougeL"], global_step=current_step)
        # save the model
        model.save_pretrained(f"./image-captioning/checkpoint-{current_step}")
        tokenizer.save_pretrained(f"./image-captioning/checkpoint-{current_step}")
        image_processor.save_pretrained(f"./image-captioning/checkpoint-{current_step}")
        # get the model back to train mode
        model.train()
        # reset the train and valid loss
        train_loss, valid_loss = 0, 0

      pixel_values = batch["pixel_values"]
      labels = batch["labels"]
      
      outputs = model(pixel_values=pixel_values, labels=labels)
      
      loss = outputs.loss
      
      loss.backward()
      
      optimizer.step()
      
      optimizer.zero_grad()
    
      loss_v = loss.item()
      train_loss += loss_v
      
      current_step += 1
      
      summary_writer.add_scalar("train_loss", loss_v, global_step=current_step)

# load the best model, change the checkpoint number to the best checkpoint
# if the last checkpoint is the best, then ignore this cell
best_checkpoint = 3000
best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)
