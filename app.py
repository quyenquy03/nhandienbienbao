from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH ='model.h5'

model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNo):
    signs = {
        0: {
            "name": "Tốc độ tối đa 20 km/h",
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 20 km/h là biển báo giới hạn tốc độ cho phép tối đa là 20 km/h. Loại biển này thường được đặt ở những khu vực có yêu cầu giảm tốc độ để đảm bảo an toàn giao thông, ví dụ như khu vực trường học, công trường, hoặc các đoạn đường hẹp, đông người qua lại."
        },
        1: {
            "name": "Tốc độ tối đa 30 km/h",
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 30 km/h là biển báo giới hạn tốc độ cho phép tối đa là 30 km/h. Biển báo này thường được đặt ở các khu vực cần kiểm soát tốc độ chặt chẽ để đảm bảo an toàn, như khu dân cư, gần trường học, hoặc những đoạn đường nguy hiểm."
        },
        2: {
            "name": "Tốc độ tối đa 50 km/h", 
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 50 km/h là biển báo giới hạn tốc độ cho phép tối đa là 50 km/h. Biển báo này thường được đặt trên các đoạn đường cần kiểm soát tốc độ trung bình để đảm bảo an toàn giao thông, chẳng hạn như trong khu dân cư hoặc những đoạn đường có mật độ xe cộ cao."
        },
        3: {
            "name": "Tốc độ tối đa 60 km/h", 
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 60 km/h là biển báo giới hạn tốc độ cho phép tối đa là 60 km/h. Biển báo này thường được đặt ở các tuyến đường mà tốc độ cần được kiểm soát vừa phải, như các tuyến đường ngoài khu dân cư hoặc trên các đường giao thông có mật độ xe trung bình."
        },
        4: {
            "name": "Tốc độ tối đa 70 km/h",
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 70 km/h là biển báo giới hạn tốc độ cho phép tối đa là 70 km/h. Biển này thường được đặt trên các tuyến đường lớn, nơi có lưu lượng xe cao hơn."
        },
        5: {
            "name": "Tốc độ tối đa 80 km/h",
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 80 km/h là biển báo giới hạn tốc độ cho phép tối đa là 80 km/h. Biển báo này thường được đặt trên các tuyến đường quốc lộ hoặc đường cao tốc."
        },
        6: {
            "name": "Hết tốc độ tối đa 80 km/h",
            "type": "Biển hết giới hạn tốc độ",
            "description": "Biển báo này thông báo rằng người lái xe đã hết giới hạn tốc độ 80 km/h và có thể tăng tốc."
        },
        7: {
            "name": "Tốc độ tối đa 100 km/h",
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 100 km/h cho phép tốc độ tối đa là 100 km/h, thường được đặt trên các tuyến đường cao tốc."
        },
        8: {
            "name": "Tốc độ tối đa 120 km/h",
            "type": "Biển giới hạn tốc độ",
            "description": "Biển báo tốc độ tối đa 120 km/h cho phép tốc độ tối đa là 120 km/h, thường được đặt trên các tuyến đường cao tốc."
        },
        9: {
            "name": "Cấm vượt",
            "type": "Biển cấm",
            "description": "Biển báo cấm vượt cho biết rằng việc vượt xe tại khu vực này là không được phép."
        },
        10: {
            "name": "Cấm vượt xe trên 3,5 tấn",
            "type": "Biển cấm",
            "description": "Biển báo này cấm vượt đối với các phương tiện có trọng tải trên 3,5 tấn."
        },
        11: {
            "name": "Quyền ưu tiên tại ngã tư tiếp theo",
            "type": "Biển ưu tiên",
            "description": "Biển báo này thông báo rằng xe ở hướng có biển báo sẽ có quyền ưu tiên tại ngã tư tiếp theo."
        },
        12: {
            "name": "Đường ưu tiên",
            "type": "Biển ưu tiên",
            "description": "Biển báo này thông báo rằng đây là đường ưu tiên, phương tiện đi trên đường này không phải nhường đường."
        },
        13: {
            "name": "Nhường đường",
            "type": "Biển nhường đường",
            "description": "Biển báo này yêu cầu các phương tiện phải nhường đường cho các phương tiện khác."
        },
        14: {
            "name": "Dừng lại",
            "type": "Biển dừng",
            "description": "Biển báo yêu cầu các phương tiện phải dừng lại trước khi tiếp tục di chuyển."
        },
        15: {
            "name": "Cấm xe",
            "type": "Biển cấm",
            "description": "Biển báo này cấm tất cả các phương tiện không được phép đi vào khu vực này."
        },
        16: {
            "name": "Cấm xe trên 3,5 tấn",
            "type": "Biển cấm",
            "description": "Biển báo này cấm các phương tiện có trọng tải trên 3,5 tấn vào khu vực này."
        },
        17: {
            "name": "Cấm vào",
            "type": "Biển cấm",
            "description": "Biển báo này thông báo rằng không được phép vào khu vực này."
        },
        18: {
            "name": "Cảnh báo chung",
            "type": "Biển cảnh báo",
            "description": "Biển báo này đưa ra cảnh báo chung về tình hình giao thông có thể xảy ra rủi ro."
        },
        19: {
            "name": "Đường cong nguy hiểm bên trái",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo có đường cong nguy hiểm bên trái."
        },
        20: {
            "name": "Đường cong nguy hiểm bên phải",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo có đường cong nguy hiểm bên phải."
        },
        21: {
            "name": "Đường cong kép",
            "type": "Biển cảnh báo",
            "description": "Biển báo này thông báo có hai đường cong liên tiếp."
        },
        22: {
            "name": "Đường gồ ghề",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo rằng mặt đường có thể gồ ghề."
        },
        23: {
            "name": "Đường trơn trượt",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo rằng mặt đường có thể trơn trượt, đặc biệt trong thời tiết xấu."
        },
        24: {
            "name": "Đường hẹp bên phải",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo rằng đường sẽ hẹp lại bên phải."
        },
        25: {
            "name": "Công trường thi công",
            "type": "Biển cảnh báo",
            "description": "Biển báo này thông báo có công trường thi công trong khu vực này."
        },
        26: {
            "name": "Tín hiệu giao thông",
            "type": "Biển tín hiệu",
            "description": "Biển báo này chỉ ra rằng có tín hiệu giao thông cần phải tuân theo."
        },
        27: {
            "name": "Người đi bộ",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo có người đi bộ qua đường."
        },
        28: {
            "name": "Trẻ em qua đường",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo có trẻ em qua đường, cần chú ý."
        },
        29: {
            "name": "Xe đạp qua đường",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo có xe đạp qua đường."
        },
        30: {
            "name": "Cẩn thận băng tuyết",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo về khả năng có băng tuyết trên đường."
        },
        31: {
            "name": "Động vật hoang dã qua đường",
            "type": "Biển cảnh báo",
            "description": "Biển báo này cảnh báo có thể có động vật hoang dã qua đường."
        },
        32: {
            "name": "Hết tất cả giới hạn tốc độ và cấm vượt",
            "type": "Biển hết giới hạn",
            "description": "Biển báo này thông báo hết tất cả các giới hạn tốc độ và cấm vượt trước đó."
        },
        33: {
            "name": "Rẽ phải phía trước",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe sẽ phải rẽ phải phía trước."
        },
        34: {
            "name": "Rẽ trái phía trước",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe sẽ phải rẽ trái phía trước."
        },
        35: {
            "name": "Chỉ đi thẳng",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe chỉ có thể đi thẳng."
        },
        36: {
            "name": "Đi thẳng hoặc rẽ phải",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe có thể đi thẳng hoặc rẽ phải."
        },
        37: {
            "name": "Đi thẳng hoặc rẽ trái",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe có thể đi thẳng hoặc rẽ trái."
        },
        38: {
            "name": "Giữ bên phải",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe phải giữ bên phải."
        },
        39: {
            "name": "Giữ bên trái",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng người lái xe phải giữ bên trái."
        },
        40: {
            "name": "Bắt buộc rẽ vòng",
            "type": "Biển chỉ dẫn",
            "description": "Biển báo này chỉ dẫn rằng phải rẽ vòng tại khu vực này."
        },
        41: {
            "name": "Hết cấm vượt",
            "type": "Biển hết cấm",
            "description": "Biển báo này thông báo rằng đã hết cấm vượt."
        },
        42: {
            "name": "Hết cấm vượt xe trên 3,5 tấn",
            "type": "Biển hết cấm",
            "description": "Biển báo này thông báo rằng đã hết cấm vượt đối với các phương tiện có trọng tải trên 3,5 tấn."
        }
    }
    # Trả về object gồm tên, loại và mô tả, hoặc giá trị mặc định nếu không tìm thấy
    return signs.get(classNo, {"name": "Biển báo không xác định", "type": "Không xác định", "description": "Biển báo không xác định"})

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Original Image", img)
    img = cv2.resize(img, (32, 32))

    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    # PREDICT IMAGE
    predictions = model.predict(img)
    print(predictions)
    classIndex = np.argmax(predictions, axis=-1)  # Lấy chỉ số lớp với xác suất cao nhất
    print(classIndex)
    preds = getClassName(classIndex[0])  # Chuyển đổi chỉ số lớp thành tên lớp
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
