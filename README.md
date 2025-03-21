# Text-to-Image Generation with Latent Diffusion Models

## Project Overview

This project implements a text-to-image generation system using Latent Diffusion Models, specifically focusing on fine-tuning the CLIP text encoder while keeping other components frozen. The implementation uses the Stable Diffusion v1.5 architecture and is optimized to work on CPU environments.

## Architecture Components

The system is composed of three main components:

1. **Variational Autoencoder (VAE)**
   - Uses AutoencoderKL from Stable Diffusion v1.5
   - 8x downsampling with 4 channels
   - Gaussian prior distribution
   - Kept frozen during training

2. **U-Net Model**
   - UNet2DConditionModel from Stable Diffusion v1.5
   - Performs latent diffusion
   - Features cross-attention mechanisms and skip connections
   - Kept frozen during training

3. **CLIP Text Encoder**
   - Based on clip-vit-base-patch32
   - Vision component: 768 dimensions, 32x32 patches
   - Text component: 512 dimensions, 12 layers
   - Fine-tuned on the Flickr8k dataset

## Implementation Details

### Training Pipeline

The training process consists of:

1. **Data Preparation**
   - Loading and preprocessing images
   - Processing text captions
   - Creating batches with proper padding
   - Applying CLIP-specific preprocessing

2. **Training Loop**
   - Mixed precision training (fp16)
   - Gradient accumulation
   - Regular checkpointing

3. **Optimization Strategy**
   - AdamW optimizer
   - Learning rate: 1e-5
   - Gradient clipping at 1.0
   - Batch size: 32

### Fine-Tuning Strategy

The project uses a selective fine-tuning approach:
- Only the CLIP text encoder is fine-tuned
- The VAE and U-Net components remain frozen with pre-trained weights
- The Flickr8k dataset is used for training
- Data flows through the CLIP processor before entering the models

## Image Generation Process

The generation process follows three main steps:

1. **Input Processing**
   - Text prompt is converted to embeddings by the fine-tuned CLIP model
   - Random noise is generated as a starting point

2. **Guided Denoising**
   - U-Net gradually refines the random noise into meaningful content
   - Text embeddings guide the denoising process to match the description

3. **Final Rendering**
   - VAE decoder transforms the refined latent representation into a high-quality image
   - The output aligns with the original text prompt

## Results

The training showed successful convergence with:
- Training loss per batch showing fluctuating but generally decreasing values
- Average loss per epoch decreasing smoothly from 0.20 to approximately 0.06 over 9 epochs

The model demonstrates the ability to generate high-quality images from text prompts, including complex scenes like "A dog running on the beach at sunset" and "A colorful garden with blooming flowers."

## Web Application

The project includes a Flask web application (`app.py`) that serves the model:

- Provides a web interface for users to enter text prompts
- Optimized for CPU inference with appropriate configurations
- Implements memory-efficient settings like attention slicing
- Renders and returns generated images in PNG format

## Files in the Repository

- `app.py`: Flask web application for serving the model
- `Latent_Diffusion_Model.ipynb`: Jupyter notebook with model implementation and training code
- `final_model/`: Directory containing saved model weights
  - `final_weights.pt`: Trained model weights

## Usage

### Requirements

```
torch
diffusers
transformers
flask
pillow
```

### Running the Web Application

```bash
python app.py
```
Then navigate to http://localhost:5000 in your web browser.

### Using the Jupyter Notebook

The `Latent_Diffusion_Model.ipynb` notebook contains the full implementation, including:
- Model architecture setup
- Training pipeline implementation
- Visualization of training results
- Image generation examples

## Future Work

Potential areas for improvement:
- Extend fine-tuning to other components of the model
- Explore different datasets beyond Flickr8k
- Implement additional optimization techniques for faster CPU inference
- Add more control parameters for guided image generation