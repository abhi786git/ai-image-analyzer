from flask import Flask, render_template, request, send_file, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import io
from PIL import Image, ImageStat
import numpy as np
from collections import Counter
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.json_encoder = NumpyEncoder  # Use custom JSON encoder for Flask responses

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load BLIP model for captioning
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Generate caption function
def generate_caption(image_path):
    """Generate AI-based caption using BLIP"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ------------------- IMAGE PROCESSING FUNCTION -------------------
def analyze_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    # Get image properties
    results = {
        "filename": os.path.basename(image_path),
        "dimensions": f"{image.width} x {image.height}",
        "color_type": image.mode,
        "channels": len(image.getbands()),
        "file_size": f"{os.path.getsize(image_path) / 1024:.2f} KB",
    }

    # Calculate Brightness
    stat = ImageStat.Stat(image.convert("L"))
    results["brightness"] = round(stat.mean[0], 2)

    # Calculate Contrast
    results["contrast"] = round(stat.stddev[0], 2)

    # Calculate Sharpness (Using Variance of Laplacian)
    gray = image.convert("L")
    gray_array = np.array(gray)
    results["sharpness"] = round(np.var(gray_array), 2)

    # Get 5 Dominant Colors
    pixels = image_array.reshape(-1, image_array.shape[-1])
    pixel_list = [tuple(pixel) for pixel in pixels]
    
    # Convert NumPy values to regular Python integers
    clean_pixel_list = []
    for pixel in pixel_list:
        clean_pixel = tuple(int(value) for value in pixel)
        clean_pixel_list.append(clean_pixel)
    
    most_common_colors = Counter(clean_pixel_list).most_common(5)
    results["dominant_colors"] = [color[0] for color in most_common_colors]
    
    # Generate caption
    try:
        results["caption"] = generate_caption(image_path)
    except Exception as e:
        results["caption"] = f"Caption generation failed: {str(e)}"

    return results


# ------------------- ROUTES -------------------
@app.route("/array")
def get_array():
    arr = np.array([1, 2, 3, 4], dtype=np.uint8)
    return jsonify({"values": arr.tolist()})  # Convert NumPy array to list

@app.route("/data")
def get_data():
    num = np.uint8(255)  # NumPy uint8 value
    return jsonify({"value": int(num)})  # Convert uint8 to int before returning

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "Error: No file uploaded. Please select an image."

        file = request.files["file"]
        if file.filename == "":
            return "Error: No selected file."

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            results = analyze_image(file_path)
            session["results"] = results
            session["file_path"] = file_path

            return redirect(url_for("results"))

    return render_template("index.html")


@app.route("/results")
def results():
    results = session.get("results", None)
    file_path = session.get("file_path", None)

    if not results or not file_path:
        return redirect(url_for("home"))

    # Convert file path for template
    relative_path = "/" + file_path if not file_path.startswith("/") else file_path
    
    return render_template("result.html", results=results, file_path=relative_path)


@app.route("/caption_only", methods=["POST"])
def caption_only():
    """Endpoint for just generating captions"""
    if "file" not in request.files:
        return "Error: No file uploaded. Please select an image."

    file = request.files["file"]
    if file.filename == "":
        return "Error: No selected file."

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Generate caption
        caption = generate_caption(file_path)
        
        # Store in session
        session["image_path"] = file_path
        session["caption"] = caption
        
        return redirect(url_for("display_caption"))


@app.route("/display_caption")
def display_caption():
    """Display only the caption results"""
    if "image_path" not in session:
        return redirect(url_for("home"))
        
    image_path = session["image_path"]
    caption = session["caption"]
    
    # Convert file path for template
    relative_path = "/" + image_path if not image_path.startswith("/") else image_path
    
    return render_template("display.html", image_path=relative_path, caption=caption)


@app.route("/download_report")
def download_report():
    results = session.get("results", None)
    if not results:
        return "No results to download"

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Set Title
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(100, 750, " AI Image Analysis Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Line Separator
    pdf.setStrokeColor(colors.black)
    pdf.line(100, 720, 500, 720)

    # Image Caption Section
    y_position = 700
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y_position, " AI Generated Caption:")
    
    y_position -= 20
    # Using standard Helvetica instead of Helvetica-Italic which might not be available
    pdf.setFont("Helvetica", 12)
    
    # Handle potentially long captions with text wrapping
    caption = results.get("caption", "No caption available")
    text_obj = pdf.beginText(120, y_position)
    text_obj.setFont("Helvetica", 12)  # Use regular Helvetica instead of Italic
    
    # Simple text wrapping for the caption
    max_width = 400
    words = caption.split()
    line = ""
    
    for word in words:
        test_line = line + " " + word if line else word
        if pdf.stringWidth(test_line, "Helvetica", 12) < max_width:
            line = test_line
        else:
            text_obj.textLine(line)
            line = word
    
    if line:
        text_obj.textLine(line)
    
    pdf.drawText(text_obj)
    
    # Calculate new y_position after caption
    y_position = text_obj.getY() - 20

    # Image Properties Section
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y_position, " Image Properties:")

    pdf.setFont("Helvetica", 12)
    data_entries = [
        (" Filename:", results["filename"]),
        (" Dimensions:", results["dimensions"]),
        (" Color Type:", results["color_type"]),
        (" Channels:", results["channels"]),
        (" File Size:", results["file_size"]),
        (" Brightness:", results["brightness"]),
        (" Contrast:", results["contrast"]),
        (" Sharpness:", results["sharpness"]),
    ]

    for label, value in data_entries:
        y_position -= 20
        pdf.drawString(120, y_position, f"{label} {value}")

    # Dominant Colors Section
    y_position -= 30
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y_position, " Top 5 Dominant Colors:")

    pdf.setFont("Helvetica", 12)
    for color in results["dominant_colors"]:
        y_position -= 20
        pdf.drawString(120, y_position, f"RGB: {color}")

    # Footer
    pdf.setFont("Helvetica", 10)  # Use regular Helvetica instead of Oblique
    pdf.setFillColor(colors.grey)
    pdf.drawString(100, 50, "Generated by AI Image Feature Analyzer")

    # Save & Send File
    pdf.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name="AI_Image_Report.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
