from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import requests
import json
import uuid
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part

load_dotenv()
app = Flask(__name__)
Gemini_Api_Key = os.getenv("Gemini_Api_Key")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")  # Your Google Cloud project ID

# Configure Gemini
genai.configure(api_key=Gemini_Api_Key)

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID)

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
            "Create a detailed image description for an AI image generator based on this prompt. " +
            "Focus on visual details, style, lighting, and composition. Keep it under 100 words: " + prompt
        ])
        enhanced_prompt = enhanced_prompt_response.text
        
        # Step 2: Generate the image using Vertex AI's Imagen
        imagen_model = GenerativeModel("imagegeneration@002")
        image_response = imagen_model.generate_content(
            enhanced_prompt,
            generation_config={
                "temperature": 0.9,
            },
        )
        
        # Extract the image
        image_bytes = image_response.candidates[0].content.parts[0].inlined_data.data
        
        # Save the image to a file with a unique name
        image_filename = f"static/generated_images/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(image_filename), exist_ok=True)
        
        with open(image_filename, "wb") as f:
            f.write(base64.b64decode(image_bytes))
        
        # Return the image URL
        image_url = "/" + image_filename
        
        return jsonify({
            "image_url": image_url,
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt
        }), 200
        
    except Exception as e:
        print('error_msg:', e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)