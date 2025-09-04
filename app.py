import base64
import cv2
import numpy as np

from flask import Flask, jsonify, request
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

app = Flask(__name__)

THRESHOLD = 200

def preprocess_image_buffer(buf):
    # unblur the image
    nparr = np.frombuffer(buf, np.uint8)
    imbuf = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def variance_of_laplacian(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def unblur(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    focus_measure = variance_of_laplacian(imbuf)
    iterations = 0

    while focus_measure < THRESHOLD:
        imbuf = unblur(imbuf)
        iterations += 1
        focus_measure = variance_of_laplacian(imbuf)

    return imbuf


@app.route('/extract', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({
            'code': 400,
            'message': 'Please upload at least one image.'
        }), 400

    imbuf = preprocess_image_buffer(request.files['image'].read())
    results = ocr.predict(imbuf)

    for out in results:
        print(out['rec_texts'])

    return jsonify({'code': 200, 'message': 'All OK'}), 200
