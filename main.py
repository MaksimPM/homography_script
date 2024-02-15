import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import imageio.v2 as imageio


def select_landmarks(image):
    if image is None:
        print("Ошибка: Изображение не было загружено.")
        return None

    """Масштабирование изображения для предотвращения выхода за границы экрана"""
    max_height = 600
    max_width = 800
    scale_factor = min(max_height / image.shape[0], max_width / image.shape[1])
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    """Отображение изображения и выбор опорных точек"""
    clone = scaled_image.copy()
    points = []

    def select_points(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", clone)
            points.append([int(x / scale_factor), int(y / scale_factor)])

    cv2.imshow("Image", clone)
    cv2.setMouseCallback("Image", select_points)

    cv2.waitKey(0)

    return np.array(points)


def read_16bit_tiff(filepath):
    image = imageio.imread(filepath)
    image_8bit = (image / np.max(image) * 255).astype(np.uint8)
    return image_8bit


def analyze_images(image_A, image_B):
    """Выбор опорных точек на каждом изображении"""
    landmarks_A = select_landmarks(image_A)
    landmarks_B = select_landmarks(image_B)

    if landmarks_A is None or landmarks_B is None:
        return None

    try:
        """Вычисление гомографии методом RANSAC"""
        H, _ = cv2.findHomography(landmarks_A, landmarks_B, cv2.RANSAC)
    except cv2.error as e:
        print("Ошибка:", e)
        print("Недостаточно точек для вычисления гомографии.")
        return None

    print("Гомография H:")
    print(H)

    """Применение гомографии к изображению A"""
    warped_image_A = cv2.warpPerspective(image_A, H, (image_B.shape[1], image_B.shape[0]))

    """Определение области пересечения двух изображений"""
    intersection_rect = cv2.boundingRect(np.vstack((landmarks_A, landmarks_B)))

    """Выделение области пересечения на изображениях"""
    image_A_intersection = image_A[intersection_rect[1]:intersection_rect[1] + intersection_rect[3],
                           intersection_rect[0]:intersection_rect[0] + intersection_rect[2]]
    image_B_intersection = image_B[intersection_rect[1]:intersection_rect[1] + intersection_rect[3],
                           intersection_rect[0]:intersection_rect[0] + intersection_rect[2]]

    return warped_image_A, image_A_intersection, image_B_intersection


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Homography Script")

        self.image_A = None
        self.image_B = None

        self.btn_load_image_A = tk.Button(self.root, text="Загрузить изображение A", command=self.load_image_A)
        self.btn_load_image_A.pack()

        self.btn_load_image_B = tk.Button(self.root, text="Загрузить изображение B", command=self.load_image_B)
        self.btn_load_image_B.pack()

        self.btn_process_images = tk.Button(self.root, text="Анализировать изображения", command=self.process_images)
        self.btn_process_images.pack()

    def load_image_A(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_A = read_16bit_tiff(file_path)

    def load_image_B(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_B = read_16bit_tiff(file_path)

    def process_images(self):
        if self.image_A is None or self.image_B is None:
            print("Необходимо загрузить оба изображения.")
            return

        warped_image_A, image_A_intersection, image_B_intersection = analyze_images(self.image_A, self.image_B)

        if warped_image_A is not None:
            cv2.destroyAllWindows()

            """Сохранение областей пересечения"""
            file_path_A = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])
            file_path_B = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])

            if file_path_A and file_path_B:
                cv2.imwrite(file_path_A, image_A_intersection)
                cv2.imwrite(file_path_B, image_B_intersection)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
