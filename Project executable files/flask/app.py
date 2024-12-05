from flask import Flask, Response, render_template,request,redirect, url_for, send_from_directory
import os
import cv2
import pandas as pd
from ultralytics import YOLO # type: ignore
import cvzone 
import numpy as np
app = Flask(__name__, template_folder='template')
model = YOLO('best.pt')
def generate_frames():
    count = 0
    while True:
        ret, frame = map.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020,500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        for index,row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d] # type: ignore
            cv2.rectangle(frame, (x1,y1), (x2,y2),(255,0,255),2)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) 
            _, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
@app.route('/submit',methods=["POST","GET"])
def submit():
    text = request.form['userInput']
    data = []
    text = re.sub('[^a-zA-Z]','',text.lower()) # type: ignore
    text = text.split()
    text = [ps.stem(w) for w in text if w not in set(stopwords.words('english'))] # type: ignore
    text = ''.join(text)
    data.append(text)
    x = vectorizer.transform(data).toarray() # type: ignore
    pred = model.predict(x)
    categories = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    output = categories[pred[0]]
    return render_template("output.html", results=f"The predicted news topic is: {output}")
@app.route('/')
@app.route('/home', methods=['GET','POST'])
def home():
    return render_template('home.html')
@app.route('/output')
def output():
    return render_template('output.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame') # type: ignore
if __name__ == '__main__':
    app.run(debug=True)
