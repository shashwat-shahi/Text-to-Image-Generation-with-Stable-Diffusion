from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

app = Flask(__name__)

# Initialize model with CPU configurations
def load_model():
    """Initialize model with CPU configurations"""
    device = torch.device("cpu")
    
    # Initialize pipeline with CPU settings
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32  # Use float32 for CPU
    ).to(device)
    
    # Load the saved state dict
    state_dict = torch.load(
        "final_model/final_weights.pt", 
        map_location=device,
        weights_only=True
    )
    
    # Filter text model weights
    text_model_dict = {k: v for k, v in state_dict.items() 
                      if k.startswith('text_model') and 
                      v.shape == pipe.text_encoder.state_dict()[k].shape}
    
    # Load filtered weights
    pipe.text_encoder.load_state_dict(text_model_dict, strict=False)
    
    # Enable memory efficient settings for CPU
    pipe.enable_attention_slicing(slice_size="auto")
    
    # Remove the CPU offload line since it requires CUDA
    # pipe.enable_sequential_cpu_offload()  # Remove this line
    
    return pipe

# Load model at startup
model = load_model()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        prompt = request.form['prompt']
        
        # Generate image with optimized settings for CPU
        with torch.no_grad():
            image = model(
                prompt=prompt,
                num_inference_steps=30,  # Reduced steps for faster generation
                guidance_scale=7.5
            ).images[0]
        
        # Convert to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False for production