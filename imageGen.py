from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import base64
import uuid
from PIL import Image
from io import BytesIO

load_dotenv()
app = Flask(__name__)
Gemini_Api_Key = os.getenv("Gemini_Api_Key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Optional, can work without it

# Configure Gemini
genai.configure(api_key=Gemini_Api_Key)

@app.route("/")
def home():
    return render_template("chatPlusImage.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([user_input])
        reply = response.text
        return jsonify({"reply": reply}), 200

    except Exception as e:
        print('error_msg:', e)
        return jsonify({"error": str(e)}), 500

@app.route("/generate-image", methods=["POST"])
def generate_image():
    prompt = request.json.get("prompt")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Step 1: Use Gemini to enhance the image prompt
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        enhanced_prompt_response = gemini_model.generate_content([
            "Create a detailed, artistic image description for an AI image generator based on this prompt. " +
            "Focus on visual details, style, lighting, composition, and artistic elements. " +
            "Make it descriptive but keep it under 77 words (API limit): " + prompt
        ])
        enhanced_prompt = enhanced_prompt_response.text
        
        # Step 2: Generate image using Hugging Face API
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        
        headers = {}
        if HUGGINGFACE_API_KEY:
            headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            # Create directory if it doesn't exist
            os.makedirs("static/generated_images", exist_ok=True)
            
            # Save the image
            image_filename = f"static/generated_images/{uuid.uuid4()}.png"
            
            # Convert response to image and save
            image = Image.open(BytesIO(response.content))
            image.save(image_filename)
            
            # Return the image URL
            image_url = "/" + image_filename
            
            return jsonify({
                "image_url": image_url,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt
            }), 200
        
        elif response.status_code == 503:
            return jsonify({
                "error": "Image generation model is currently loading. Please try again in a few moments."
            }), 503
        
        else:
            return jsonify({
                "error": f"Image generation failed: {response.text}"
            }), response.status_code
        
    except Exception as e:
        print('error_msg:', e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)