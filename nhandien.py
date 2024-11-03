import cv2
import numpy as np

# Đọc ảnh ban đầu
img = cv2.imread('image.png')
cv2.namedWindow('TrackbarWindow')

def getLargestContour(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_area = 0
    largest_contour = None

    # Tìm đường viền lớn nhất
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_contour = cnt

    # Vẽ đường viền lớn nhất và cắt vùng lớn nhất
    if largest_contour is not None and largest_area > 1000:  # Kiểm tra nếu có đường viền lớn hơn 1000
        cv2.drawContours(imgContour, largest_contour, -1, (255, 0, 0), 1)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Cắt vùng lớn nhất từ ảnh gốc
        cropped_img = img[y:y + h, x:x + w]
        return cropped_img  # Trả về vùng cắt

    return None  # Nếu không tìm thấy vùng lớn hơn 1000

while True:
    imgCountour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 100, 180)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, (7, 7), iterations=1)

    # Lấy vùng lớn nhất
    cropped_img = getLargestContour(imgDil, imgCountour)

    # Hiển thị các ảnh
    cv2.imshow("Original", imgCountour)
    cv2.imshow("TrackbarWindow", imgCanny)

    if cropped_img is not None:
        cv2.imshow("Cropped Largest Area", cropped_img)  # Hiển thị vùng cắt

    # Nhấn 's' để lưu ảnh cắt, nhấn 'q' để thoát
    key = cv2.waitKey(1)
    if key == ord('s') and cropped_img is not None:
        cv2.imwrite("largest_area.png", cropped_img)  # Lưu vùng cắt
        print("Vùng cắt đã được lưu thành 'largest_area.png'")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
