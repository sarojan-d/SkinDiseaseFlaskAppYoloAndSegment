import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from detect import detect_image  # Import function directly

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "uploads")
RESULTS_FOLDER = os.path.join(os.getcwd(), "static", "results")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Run YOLO detection **without subprocess**
    result_path, detections = detect_image(filepath)

    return jsonify({"output_image": f"/static/results/{file.filename}", "detections": detections})

@app.route("/static/results/<filename>")
def get_result(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
