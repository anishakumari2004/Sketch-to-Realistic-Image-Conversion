from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from sketch2real import generate  # Ensure this file is in the same directory

app = Flask(__name__)
CORS(app)

# Use absolute paths to prevent "file not found" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Sketch2Real API is active."

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        if "sketch" not in request.files:
            return jsonify({"error": "No sketch uploaded"}), 400

        file = request.files["sketch"]
        prompt = request.form.get("prompt", "a realistic photo")

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # Create unique filename
        file_id = str(uuid.uuid4())
        sketch_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.png")
        output_filename = f"{file_id}_result.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Save the uploaded sketch
        file.save(sketch_path)
        print(f"--- Processing: {prompt} ---")

        # Call your AI generation function
        # Note: device='cpu' will be slow; watch your terminal for progress
        generate(
            sketch_path=sketch_path,
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, watermark",
            output_path=output_path,
            steps=20,
            guidance=7.5,
            controlnet_strength=0.9,
            seed=-1,
            device="cpu",
            low_vram=False,
            size=512
        )

        # Construct the URL pointing to our custom route
        image_url = f"http://127.0.0.1:5000/outputs/{output_filename}"
        print(f"--- Success: {image_url} ---")

        return jsonify({"image_url": image_url})

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

# This route serves the actual image files to the browser
@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)
if __name__ == "__main__":
    print("🚀 Starting Sketch2Real API...")
    
    app.run(
        host="127.0.0.1", 
        port=5000, 
        debug=True,
        use_reloader=False  # <--- ADD THIS LINE
    )