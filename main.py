import cv2
import numpy as np


def flood_fill(img):
    # Copy ảnh để flood fill
    im_floodfill = img.copy()

    # Tạo mask (phải lớn hơn 2 pixel so với ảnh)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill từ góc (0,0) → nền
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Đảo ảnh flood fill
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Kết hợp để có ảnh đã lấp lỗ
    img_filled = img | im_floodfill_inv

    return img_filled


# ==========START=============

kernel3 = np.ones((3, 3), np.uint8)

# Step 1: Đọc ảnh -> convert thành gray
img = cv2.imread("./img/2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Làm mờ giảm nhiễu: median filter -> gauss filter
blur = cv2.medianBlur(gray, 5)
blur = cv2.GaussianBlur(blur, (5, 5), 0)

# Step 3: Thực hiện erosion (3 lần) -> dilate (2 lần) -> erosion
# => erosion nhiều để giảm nhiễu và để tách các hạt gạo (giảm số hạt gạo dính vào nhau)
img3 = blur
img3 = cv2.erode(img3, kernel3)
img3 = cv2.erode(img3, kernel3)
img3 = cv2.erode(img3, kernel3)
img3 = cv2.dilate(img3, kernel3)
img3 = cv2.dilate(img3, kernel3)
img3 = cv2.erode(img3, kernel3)

# Step 4: Tìm biên laplace -> tăng tương phản ảnh bằng clahe
# -> Nhị phân hóa theo ngưỡng otsu -> -> closing -> flood fill
laplacian = cv2.Laplacian(img3, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(laplacian)

# Nhị phân hóa Otsu
_, thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Closing: Mục đích là nối những biên của hạt gạo chưa khép kín
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel3, iterations=1)

# Fill tất cả những đường biên khép kín thành màu trắng
thresh_filled = flood_fill(thresh)

# Erosion để loại nhiễu
thresh_clean = thresh_filled

thresh_clean = cv2.erode(thresh_clean, kernel3)
thresh_clean = cv2.erode(thresh_clean, kernel3)
thresh_clean = cv2.dilate(thresh_clean, kernel3)

# Tìm contours
contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 10  # vùng nào dưới 10px -> nhiễu -> ko đếm
rice_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
print("Số hạt gạo đếm được:", len(rice_contours))

cv2.imwrite("2_blur.png", blur)
cv2.imwrite("3_preprocess.png", img3)
cv2.imwrite("4_1_laplace.png", laplacian)
cv2.imwrite("4_2_clahe_img.png", clahe_img)
cv2.imwrite("4_3_thresh.png", thresh)
cv2.imwrite("4_4_thresh_filled.png", thresh_filled)
cv2.imwrite("4_5_thresh_clean.png", thresh_clean)

# Vẽ contour lên ảnh
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

cv2.imshow("Rice Counter", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
