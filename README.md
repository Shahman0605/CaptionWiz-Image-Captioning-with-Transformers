# Image Captioning with Hugging Face Transformers

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains code for training, fine-tuning, and evaluating image captioning models using Hugging Face Transformers. The code includes an image captioning model, training, evaluation loops, and an inference script for generating image captions.

## Table of Contents

- [Getting Started](#getting-started)
- [Training and Fine-Tuning](#training-and-fine-tuning)
    - [Training a Model](#training-a-model)
    - [Fine-Tuning a Pretrained Model](#fine-tuning-a-pretrained-model)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [Note](#note)

## Getting Started

Before using this code, ensure you have the required dependencies installed:

```bash
pip install torch transformers pillow tqdm requests datasets
```
## Training and Fine-Tuning

### Training a Model

To train an image captioning model:

1. Load a fine-tuned image captioning model, tokenizer, and image processor.
2. Define the image captioning model's architecture.
3. Load the COCO dataset and preprocess it.
4. Train the model using a DataLoader and an optimizer.
5. Log training statistics, validation metrics, and save checkpoints.

### Fine-Tuning a Pretrained Model

Alternatively, fine-tune a pretrained model using the nlpconnect/vit-gpt2-image-captioning model and its associated tokenizer and image processor. Customize the dataset and fine-tuning process.

## Evaluation

After training or fine-tuning, you can evaluate your image captioning model using various metrics, including ROUGE and BLEU. The code includes functions for computing these metrics and allows you to evaluate the model on a test dataset. You can customize the evaluation metrics and specify the number of testing steps.

The code snippet below demonstrates the process of evaluating the image captioning model. It includes the following steps:

1. Load a pretrained or fine-tuned model for evaluation.
2. Define the evaluation metrics, including ROUGE and BLEU.
3. Get the evaluation metrics using the `get_evaluation_metrics` function.
4. Print the evaluation results, including ROUGE scores, BLEU scores, and test loss.

```python
# Load a pretrained or fine-tuned model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model).to(device)

# Define the evaluation metrics
metrics = get_evaluation_metrics(model, test_dataset)

# Print the evaluation results
print(metrics)

# Compare with another model
image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning")
image_captioner.model = image_captioner.model.to(device)

# Get evaluation metrics for the other model
get_evaluation_metrics(image_captioner.model, test_dataset)
```
## Inference

The repository provides an inference script that allows you to generate image captions using trained models. Follow these steps to generate captions:

1. Load a pretrained model for image captioning.
2. Define the model's architecture and tokenizer.
3. Choose an image URL for which you want to generate captions.
4. Use the provided `show_image_and_captions` function to display the image and generate captions using various models, including:
   - The fine-tuned model
   - The Abdou/vit-swin-base-224-gpt2-image-captioning model
   - Your own trained model

The following code snippet illustrates the inference process:

```python
# Define a function to show an image and generate captions
def show_image_and_captions(url):
  # Get the image and display it
  display(load_image(url))
  # Generate captions using various models
  our_caption = get_caption(best_model, image_processor, tokenizer, url)
  finetuned_caption = get_caption(finetuned_model, finetuned_image_processor, finetuned_tokenizer, url)
  pipeline_caption = get_caption(image_captioner.model, image_processor, tokenizer, url)
  # Print the captions
  print(f"Our caption: {our_caption}")
  print(f"nlpconnect/vit-gpt2-image-captioning caption: {finetuned_caption}")
  print(f"Abdou/vit-swin-base-224-gpt2-image-captioning caption: {pipeline_caption}")
```
This script provides an easy way to visualize and compare image captions generated by different models.

```python
# FastAPI application for finding similar images and text
# Create a cache for storing previous inference results
cache = {}

@app.post("/find_similar")
async def find_similar_image(file: UploadFile):
    image = Image.open(file.file)
    # Process the image and find the most similar image and text using the CLIP model
    similar_image, similar_text = find_most_similar_image_and text(best_model, image)
    
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
```
The provided Python code is a FastAPI application for image and text similarity search using the CLIP (Contrastive Language-Image Pre-training) model. This application exposes two endpoints, one for uploading images and finding similar images and text and the other for retrieving cached results. Below is a detailed explanation of the code:

1. Importing Libraries:
   - The code begins by importing necessary libraries, including FastAPI, PIL (Pillow) for image processing, PyTorch, Hugging Face Transformers for CLIP, and other utilities and dependencies.

2. Setting the Device:
   - The `device` variable is used to specify whether the application should utilize a GPU ("cuda") if available or default to CPU ("cpu").

3. Creating a FastAPI App:
   - A FastAPI app is created using `FastAPI()`. This app will handle incoming HTTP requests and responses for the specified endpoints.

4. Loading the CLIP Model:
   - The CLIP model consists of an encoder for processing images and a decoder for processing text. The encoder and decoder model names are specified in `encoder_model` and `decoder_model`, respectively. These models are loaded using the Hugging Face Transformers library, and the resulting model is moved to the specified device.

5. Loading the Best Model:
   - The best checkpoint of the model is loaded using the `VisionEncoderDecoderModel` from a pretrained directory. The specific checkpoint directory is specified using `f"./image-captioning/checkpoint-{best_checkpoint}"`.

6. Cache for Storing Results:
   - A cache dictionary is created to store previous inference results. This cache is used to quickly retrieve results without re-computation.

7. Defining the `/find_similar` Endpoint:
   - This is a POST endpoint that allows users to upload an image file (`UploadFile`) for similarity search. When a file is uploaded, the image is processed using the CLIP model (`find_most_similar_image_and_text`) to find the most similar image and text. The results are then stored in the cache with the filename as the key. The response includes the similar image and text.

8. Defining the `/cached_result/{filename}` Endpoint:
   - This is a GET endpoint that enables users to retrieve cached results based on the filename. If the filename exists in the cache, the similar image and text are returned. If not found, an error message is provided.

9. Running the FastAPI Application:
   - The code block at the end checks if the script is executed directly (not imported as a module) using `if __name__ == "__main__"`. If so, the FastAPI application is run using the `uvicorn` server. The application is hosted on "0.0.0.0" (all available network interfaces) and port 8000.

This FastAPI application allows users to perform image and text similarity searches using the CLIP model. It offers a simple HTTP interface for finding similar images and text and caching the results for efficient retrieval.
## Results

The training process yields several important results and outputs:

1. **Training and Validation Metrics:**
   - The training process logs training loss, validation loss, and evaluation metrics such as ROUGE-1, ROUGE-2, and ROUGE-L. These metrics help gauge the performance of the model during training.

2. **Model Checkpoints:**
   - The code saves model checkpoints at specified intervals during training. These checkpoints are crucial for resuming training or fine-tuning the model and for later inference.

3. **Fine-Tuned Model Evaluation:**
   - After fine-tuning a model, the evaluation metrics for the fine-tuned model are computed. This assessment provides insights into how well the model has adapted to a specific task or dataset.

### Model Comparison

You have the flexibility to compare different models' performance by evaluating them using various metrics. Additionally, there is a pipeline-based approach for image captioning, featuring the "Abdou/vit-swin-base-224-gpt2-image-captioning" model. This approach simplifies image caption generation.

### Usage

Here's a brief guide on how to utilize this repository:

1. **Training:**
   - Adapt the provided code to your dataset and model architecture for training or fine-tuning a model. This step is essential for creating a model tailored to your specific requirements.

2. **Evaluation:**
   - Utilize the provided evaluation functions to assess a model's performance. You can customize the evaluation metrics and specify the number of testing steps to suit your needs.

3. **Inference:**
   - Execute the provided inference script by providing an image URL. This script generates image captions using different models, including the fine-tuned model and the "Abdou/vit-swin-base-224-gpt2-image-captioning" model. It simplifies the caption generation process.

For a deeper understanding of the code's implementation and customization, explore the Jupyter notebook and code snippets available in the repository.

### Acknowledgments

This code relies on Hugging Face Transformers and harnesses the power of various pretrained models for image captioning. Special thanks go to the Hugging Face community for their valuable contributions.

### Note

This README serves as a high-level overview of the code's functionality. For a more in-depth exploration of code implementation and customization, please refer to the code snippets and Jupyter notebook provided within the repository.

### License

This project is licensed under the MIT License. For further details, please review the LICENSE file in the repository.






