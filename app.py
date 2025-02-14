from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTION_FOLDER'] = 'runs/detect/predict'  # Consistently use 'predict' as the folder

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)

# Load the YOLO model with your specified trained weights
model = YOLO('Best_Model/train2/weights/best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request.'

        file = request.files['file']

        if file.filename == '':
            return 'No file selected.'

        if file:
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Redirect to prediction route
            return redirect(url_for('predict_image', filename=file.filename))

    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Run the prediction using the YOLO model
    results = model.predict(
        source=image_path, 
        save=True, 
        project='runs/detect', 
        name='predict', 
        imgsz=640, 
        conf=0.1, 
        exist_ok=True  # Allow overwriting the existing folder
    )

    # Check if results contain bounding boxes
    if len(results[0].boxes) == 0:
        return render_template('noresult.html')

    # Get the path to the saved image with bounding boxes
    prediction_file = os.path.basename(results[0].path)  # Extract just the filename
    
    # Redirect to result page
    return render_template('result.html', filename=prediction_file, label=results[0].names[results[0].boxes.cls[0].item()])

@app.route('/display/<path:filename>')
def display_image(filename):
    # Ensure it retrieves from the correct path
    return send_from_directory(app.config['PREDICTION_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
