from flask import Flask,render_template,request,jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
app = Flask(__name__)
Gemini_Api_Key=os.getenv("Gemini_Api_Key")

genai.configure(api_key=Gemini_Api_Key)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/chat", methods = ["POST"])

def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400


    try :
        model=genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([user_input])
        reply = response.text
        return jsonify({"reply": reply}), 200

    except Exception as e:
        print('error _ msg :',e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)













