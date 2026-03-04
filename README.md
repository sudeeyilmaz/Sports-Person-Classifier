# Sports Person Classifier 🏆

This is an end-to-end machine learning and web development project that classifies images of 5 sports celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
<img width="1353" height="649" alt="Screenshot From 2026-03-04 11-39-29" src="https://github.com/user-attachments/assets/14277f05-d8d6-4013-a80e-2ac7c49c2395" />
<img width="1353" height="649" alt="image" src="https://github.com/user-attachments/assets/60a6bec4-9723-4ae6-8b5d-86c3cac4b02b" />

**💡 Key Highlight:** This project was built specifically to understand the core concepts of Machine Learning and Computer Vision. Therefore, high-level "plug-and-play" libraries like `face-recognition` or cloud APIs were **not used**. The entire pipeline—from data cleaning and feature extraction to model training—was built from the ground up using traditional techniques.

## 🧠 How It Works (Custom ML Pipeline)

Instead of relying on deep learning black boxes, this project uses a classical, step-by-step approach:

1. **Data Cleaning & Face Detection:** Uploaded images are scanned using OpenCV Haar Cascades. To ensure model accuracy, the system only crops and processes the Region of Interest (ROI)—specifically, a clear face where at least two eyes are clearly visible.
2. **Feature Engineering:** The cropped facial images undergo a Wavelet Transform using the `PyWavelets` library. This technique helps extract key facial features, edges, and textures while filtering out unnecessary background noise.
3. **Classification:** The raw image array and the wavelet-transformed image array are vertically stacked and fed into a custom-trained Scikit-Learn machine learning model to predict the identity of the athlete.

## 🛠️ Tech Stack

* **Machine Learning & Vision:** Python, Scikit-Learn, OpenCV, PyWavelets, Numpy, Pandas
* **Backend:** Python, Flask, Flask-CORS
* **Frontend:** HTML, CSS, JavaScript, Bootstrap, jQuery, Dropzone.js

## 📁 Project Structure

* `/model`: Contains the dataset, data cleaning logic, model training, and testing files (Jupyter Notebooks / Python scripts). The final model is exported as a saved `.pkl` file.
* `/server`: Hosts the Flask backend code (`main.py`, `util.py`) and the saved model artifacts (`/artifacts`).
* `/UI`: Contains the static HTML, CSS, and JavaScript files that make up the web interface.

## 🚀 Installation and Setup

To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```
   git clone [https://github.com/sudeeyilmaz/Sports-Person-Classifier.git](https://github.com/sudeeyilmaz/Sports-Person-Classifier.git)
   cd Sports-Person-Classifier
   ```
2. Create and activate a virtual environment.
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Navigate to the server directory and start the Flask app:
```
cd server
python main.py
```
5. Open your web browser and go to http://127.0.0.1:5000 to test the classifier.
   
