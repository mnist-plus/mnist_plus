from flask import Flask
from flask import request
import matplotlib.image as image
app = Flask(__name__)

@app.route("/",methods=["POST"])
def calc():
    raw = request.form.get('raw')
    piece = raw.split(sep=',')
    matrix=[]
    for col in range(0,28):
        matrix.append([])
    i = 0
    j = 0
    for val in piece:
        if i == 280:
            i = 0
            j = j+1
        v= 255-int(val)
        matrix[j].append([v,v,v])
        i = i+1
    image._png.write_png(matrix,'123.png')
    print(matrix)
    return "Nah."

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
    app.run(host="127.0.0.1",port=8000)

if __name__ == '__main__':
    main()