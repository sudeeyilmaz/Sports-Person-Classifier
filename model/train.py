import os
import cv2
import json
import pywt
import shutil
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

face_cascade = cv2.CascadeClassifier('model/open_cv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('model/open_cv/haarcascades/haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

if __name__ == '__main__':
    
    path_to_data = "./model/dataset/"
    path_to_cr_data = "./model/dataset/cropped/"

    img_dirs = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dirs.append(entry.path)

    cropped_image_dirs = []
    celebrity_file_names_dict = {}

    if os.path.exists(path_to_cr_data) and len(os.listdir(path_to_cr_data)) > 0:
        print("Cropped directory already exists! Skipping cropping phase and loading cleaned images...")
        for entry in os.scandir(path_to_cr_data):
            if entry.is_dir():
                celebrity_name = entry.path.split('/')[-1]
                celebrity_file_names_dict[celebrity_name] = []
                for file in os.scandir(entry.path):
                    celebrity_file_names_dict[celebrity_name].append(file.path)
    else:
        print("Detecting and cropping faces...")
        if os.path.exists(path_to_cr_data):
            shutil.rmtree(path_to_cr_data)
        os.mkdir(path_to_cr_data)
        
        for img_dir in img_dirs:
            count = 1
            celebrity_name = img_dir.split('/')[-1]
            celebrity_file_names_dict[celebrity_name] = []
            print(f"Processing: {celebrity_name}")
            
            for entry in os.scandir(img_dir):
                roi_color = get_cropped_image_if_2_eyes(entry.path)
                if roi_color is not None:
                    cropped_folder = path_to_cr_data + celebrity_name
                    if not os.path.exists(cropped_folder):
                        os.makedirs(cropped_folder)
                        cropped_image_dirs.append(cropped_folder)
                    
                    cropped_file_name = celebrity_name + str(count) + ".png"
                    cropped_file_path = cropped_folder + "/" + cropped_file_name
                    cv2.imwrite(cropped_file_path, roi_color)
                    
                    celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                    count += 1

    print("Performing wavelet transform and preparing X, y arrays...")
    class_dict = {}
    count = 0
    for celebrity_name in celebrity_file_names_dict.keys():
        class_dict[celebrity_name] = count
        count += 1 

    X = []
    y = []

    for celebrity_name, training_files in celebrity_file_names_dict.items():
        for training_image in training_files:
            img = cv2.imread(training_image)
            if img is None:
                continue
            
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(scalled_raw_img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            
            combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
            X.append(combined_img)
            y.append(class_dict[celebrity_name])

    X = np.array(X).reshape(len(X), 4096).astype(float)
    
    print("Training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    model_params = {
        'svm': {
            'model': SVC(gamma='auto', probability=True),
            'params': {
                'svc__C': [1, 10, 100, 1000],
                'svc__kernel': ['rbf', 'linear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'randomforestclassifier__n_estimators': [1, 5, 10]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(),
            'params': {
                'logisticregression__C': [1, 5, 10]
            }
        }
    }

    scores = []
    best_estimators = {}

    for algo, mp in model_params.items():
        pipe = make_pipeline(StandardScaler(), mp['model'])
        clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
        clf.fit(X_train, y_train)
        
        scores.append({
            'model': algo,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
        best_estimators[algo] = clf.best_estimator_

    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    print("\n--- GridSearchCV Results ---")
    print(df)

    print("\n--- Test Scores ---")
    print("SVM Score:", best_estimators['svm'].score(X_test, y_test))
    print("Random Forest Score:", best_estimators['random_forest'].score(X_test, y_test))
    print("Logistic Regression Score:", best_estimators['logistic_regression'].score(X_test, y_test))

    best_clf = best_estimators['svm']
    
    print("\nSaving model (saved_model.pkl) and dictionary (class_dictionary.json)...")
    
    joblib.dump(best_clf, 'saved_model.pkl')

    with open('class_dictionary.json', 'w') as f:
        f.write(json.dumps(class_dict))
        
    print("All processes completed successfully!")