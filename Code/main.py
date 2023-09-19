import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect
# from inference import model 
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        # results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        results.save(save_dir="static/")
        return redirect("static/image0.jpg")

    return render_template("index.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
