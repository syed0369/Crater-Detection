from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import os, shutil
import cv2

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder
print(upload_folder)
def getBlobImg(results, image_path):
    image = cv2.imread(image_path)
    image_copy = image.copy()
    ext = image_path[image_path.rfind('.') + 1 : ]
    for result in results[0].boxes.xyxy:  
        x1, y1, x2, y2 = map(int, result[:4])  
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 2)
        cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), cv2.FILLED)
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 1)
    image = cv2.addWeighted(image, 0.7, image_copy, 0.3, 0)
    (Image.fromarray(image)).save('./static/blob/img.' + ext)
    return './static/blob/img.' + ext

@app.route('/find-craters', methods=['POST'])
def find_craters():
    image = request.files['img']
    path = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD'], path)
    image.save(image_path)
    model = YOLO('../runs/detect/best.pt')
    shutil.rmtree('./static/results', ignore_errors = True)
    results = model.predict(source = image_path, save = True, project = 'static', name = 'results')
    img = fr'./static/results/{path}'
    blob_img = getBlobImg(results, image_path)
    return render_template('index.html', img = img, blob_img = blob_img)

@app.route('/path', methods=['POST'])
def path():
    pass

@app.route('/')
def main_page():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)
