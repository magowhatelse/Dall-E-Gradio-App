import openai
import os
import gradio as gr
from skimage import io
from PIL import Image


api_key = os.getenv("API_API_KEY")

def generate_image(prompt, quality, resolution, model, style):
    client = openai.OpenAI(
        api_key=api_key,
        
    )

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            quality=quality,
            size=resolution,
            style=style,
            n=1,  
        )
        image_url = response.data[0].url
    except Exception as e:
        return f"Error fetching image: {str(e)}", None

    try:
        image = io.imread(image_url)

        # Ensure the image is in the correct format for PIL (RGB mode)
        if image.ndim == 2:  # Grayscale
            pil_image = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:  # RGBA (remove alpha channel)
            pil_image = Image.fromarray(image[:, :, :3])
        else:
            pil_image = Image.fromarray(image)

        # Save the image locally
        file_path = "generated_image.png"
        pil_image.save(file_path)

        return pil_image, file_path

    except Exception as e:
        return f"Error processing image: {str(e)}", None

# Create Gradio Interface
interface = gr.Interface(
    generate_image,
    [
        gr.Textbox(label="Prompt"),
        gr.Dropdown(label="Quality", choices=["standard", "hd"], value="standard"),
        gr.Dropdown(label="Resolution", choices=["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"], value="1024x1024"), 
        gr.Dropdown(label="Model", choices=["dall-e-3", "dall-e-2"], value="dall-e-2"),
        gr.Dropdown(label="Style", choices=["natural", "vivid"], value="vivid")
    ],
    [   
        gr.Image(label="Generated Image"),  # Display the image
        gr.File(label="Download Image")  # Download the image
    ],
    title="AI Image Generator",
    description="Enter a prompt to generate an image using DALL-E.",
)

# Launch the app
interface.launch(server_port=7801)
