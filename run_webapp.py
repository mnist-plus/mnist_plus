from flask import Flask
from flask import request
import cv2
import struct
import pickle
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logz
import ImageSplit
from imgaug import augmenters as iaa
from data_process import read_dataset, shuffle_dataset, read_sample
from writing_board import Classifier

app = Flask(__name__)

@app.route("/",methods=["POST"])
def calc():
    raw = request.form.get('raw')
    piece = raw.split(sep=',')
    matrix=[]
    for col in range(0,280):
        matrix.append([])
    i = 0
    for val in piece:
        if i == 280:
            i = 0
        matrix[i].append(255 - int(val))
        i = i+1

    matrix = np.array(matrix).T
    matrix = np.array([matrix, matrix, matrix])
    matrix = np.transpose(matrix, [1, 2, 0])
    plt.imsave('samples/test.png', matrix)

    spliter = ImageSplit.imageSplit()
    if not os.path.isdir("samples/tmp"):
        os.mkdir("samples/tmp")
    for i in os.listdir("samples/tmp"):
        filePath = os.path.join("writingBoard/tmp",i)
        #os.remove(filePath)
    spliter.imageSplit("samples/test.png","samples/tmp/", 28)

    classifier = Classifier()
    imgs = read_sample('samples/tmp/')
    ans = classifier.predict(imgs)

    return str(ans)

@app.route("/",methods=["GET"])
def equ():
    return '''<html>

<body>
<style type="text/css">
canvas {
    border:red solid thin;
}
</style>
    <script>
        p = 0
        ev = 0

        function paint(e) {
            c = document.getElementById("equation").getContext("2d")
            c.lineWidth = 2
            c.moveTo(ev.layerX, ev.layerY)
            c.lineTo(e.layerX, e.layerY)
            c.stroke()
            ev = e
        }

        function process() {
            var arr = document.getElementById('equation').getContext('2d').getImageData(0,0,280,28).data
            var res = []
            for (var index = 0; index < arr.length; index++) {
                if (index%4 == 3) {
                    res.push(arr[index])
                }
            }
            document.getElementById('raw').value=res
        }
    </script>
<canvas id="equation" name="img" width="280" height="28" onmousedown="p=1,ev=event" onmouseup="p=0" onmouseout="p=0" onmousemove="if(p==1){paint(event)}"></canvas>

    <form method="POST" onsubmit="process()" autocomplete="off">
    <div style="position:relative; width:0px; height:0px; overflow:hidden;">
        <input type="text" name="raw" id="raw" /></div>
        <input type="submit" />
    </form>
</body>

</html>'''

def main():
    app.run(host="127.0.0.1",port=8018)

if __name__ == '__main__':
    main()