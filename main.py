from diffusers import DiffusionPipeline
import torch

# Check if GPU is available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Ghibli fine-tuned model
pipe = DiffusionPipeline.from_pretrained(
    "Yntec/Ghibli",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Move the model to the available device (GPU/CPU)
pipe = pipe.to(device)

# Define a Ghibli-style prompt
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# Generate the image
with torch.inference_mode():  # Disable gradient computation for faster inference
    image = pipe(prompt).images[0]

# Save the generated image
output_path = "ghibli_style_output.png"
image.save(output_path)
print(f"✅ Image generated and saved at: {output_path}")

