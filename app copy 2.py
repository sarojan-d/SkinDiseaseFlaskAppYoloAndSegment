import os
import subprocess
from flask import Flask, request, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "uploads")
RESULTS_FOLDER = os.path.join(os.getcwd(), "static", "results")
LABELS_FOLDER = os.path.join(RESULTS_FOLDER, "labels")  # YOLO label files
WEIGHTS_PATH = os.path.join(os.getcwd(), "yolov7.pt")  # Adjust if needed

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

# Class names (Modify based on your dataset)
#['Melanocytic nevi', 'Melanoma', 'Benign keratosis', 'Basal cell carcinoma', 'Actinic Keratoses', 'Vascular lesions', 'Dermatofibroma']
CLASS_NAMES = {
    0: "Melanocytic nevi",
    1: "Melanoma",
    2: "Benign keratosis",
    3: "Basal cell carcinoma",
    4: "Actinic Keratoses",
    5: "Vascular lesions",
    6: "Dermatofibroma"
}

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

    # Run YOLO detection
    command = [
        "python", "detect.py",
        "--weights", WEIGHTS_PATH,
        "--source", filepath,
        "--conf", "0.6",
        "--img-size", "640",
        "--device", "0",
        "--save-txt",
        "--project", "static",
        "--name", "results",
        "--exist-ok"
    ]

    try:
        subprocess.run(command, check=True)

        # Get output image path
        output_image_path = os.path.join(app.config["RESULTS_FOLDER"], file.filename)

        # Read label file if it exists
        label_filename = os.path.splitext(file.filename)[0] + ".txt"
        label_filepath = os.path.join(LABELS_FOLDER, label_filename)

        detections = []
        if os.path.exists(label_filepath):
            with open(label_filepath, "r") as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        confidence = float(parts[1])
                        disease_name = CLASS_NAMES.get(class_id, f"Unknown ({class_id})")
                        detections.append({"disease": disease_name, "confidence": confidence})

        return jsonify({"output_image": f"/static/results/{file.filename}", "detections": detections})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

@app.route("/static/results/<filename>")
def get_result(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
