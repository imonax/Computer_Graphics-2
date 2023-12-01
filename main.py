import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import cv2
from PIL import Image, ImageTk
import numpy as np


def normalize_hist(channel_array):
    def Sh(h, i):
        return sum(h[:i])

    h = np.bincount(channel_array.flatten())
    H = sum(h)
    h = h / H

    for i in range(channel_array.shape[0]):
        for j in range(channel_array.shape[1]):
            channel_array[i][j] = 255 * Sh(h, channel_array[i][j])

    return channel_array


def open_filter_image():
    global filter_img
    global filter_orig
    global filter_LoG_img
    global filter_usual_img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
    img = cv2.imread(file_path)

    red = img[:, :, 2].copy()
    blue = img[:, :, 0].copy()

    img[:, :, 0] = red
    img[:, :, 2] = blue

    filter_orig = ImageTk.PhotoImage(image=Image.fromarray(img))
    filter_usual_img = ImageTk.PhotoImage(image=Image.fromarray(low_pass_filter(img, mid)))

    filter_LoG_img = ImageTk.PhotoImage(image=Image.fromarray(low_pass_filter(img, gaussian)))

    lbl_filter_orig.config(image=filter_orig)
    lbl_filter_usual.config(image=filter_usual_img)
    lbl_filter_LoG.config(image=filter_LoG_img)


def open_contrast_image():
    global color_hist_img
    global hsv_hist_img
    global linear_contrast_img
    global contrast_orig
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
    img = cv2.imread(file_path)
    red = img[:, :, 2].copy()
    blue = img[:, :, 0].copy()

    img[:, :, 0] = red
    img[:, :, 2] = blue
    contrast_orig = ImageTk.PhotoImage(image=Image.fromarray(img))

    color_hist_arr = color_histogram_equalization(img)
    color_hist_img = ImageTk.PhotoImage(image=Image.fromarray(color_hist_arr))

    hsv_hist_arr = hsv_histogram_equalization(img)
    hsv_hist_img = ImageTk.PhotoImage(image=Image.fromarray(hsv_hist_arr))

    linear_contrast_arr = linear_contrast(img)
    linear_contrast_img = ImageTk.PhotoImage(image=Image.fromarray(linear_contrast_arr))

    lbl_contrast_orig.config(image=contrast_orig)
    lbl_color_hist.config(image=color_hist_img)
    lbl_hsv_hist.config(image=hsv_hist_img)
    lbl_linear_contrast.config(image=linear_contrast_img)


root = tk.Tk()
image = cv2.imread('C:\\Users\\User\\Downloads\\example.jpg')
filter_orig = ImageTk.PhotoImage(image=Image.fromarray(image))
contrast_orig = ImageTk.PhotoImage(image=Image.fromarray(image))


def low_pass_filter(image, kernel):
    for_pad = kernel.shape[0] // 2
    padded_image = np.pad(image, ((for_pad, for_pad), (for_pad, for_pad), (0, 0)), 'constant')

    new_image = np.zeros_like(image)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            new_image += padded_image[i:image.shape[0] + i, j:image.shape[1] + j] * kernel[i, j]

    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

mid = np.array(((1, 4, 1),
                (4, 16, 4),
                (1, 4, 1)))

gaussian = np.array(((1, 1, 1),
                     (1, 2, 1),
                     (1, 1, 1)))


def linear_contrast(image):
    minimum = np.min(image)
    maximum = np.max(image)
    result = image / (maximum - minimum) * (image - minimum)
    return result.astype(np.uint8)


def color_histogram_equalization(image):
    b, g, r = cv2.split(image)
    r_eq = normalize_hist(r)
    g_eq = normalize_hist(g)
    b_eq = normalize_hist(b)
    equalized_image = cv2.merge((r_eq, g_eq, b_eq))
    build_histogram(equalized_image)
    return equalized_image


def hsv_histogram_equalization(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v_eq = normalize_hist(v)
    equalized_hsv = cv2.merge((h, s, v_eq))
    equalized_image = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)
    build_histogram(equalized_image)
    return equalized_image


def build_histogram(image):
    """color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()


lbl_filter_txt = tk.Label(text='low pass filter')
lbl_filter_txt.grid(row=0, column=0, columnspan=2)

lbl_color_hist_txt = tk.Label(text='color histogram equation')
lbl_color_hist_txt.grid(row=0, column=2)

lbl_hsv_hist_txt = tk.Label(text='hsl(only l) histogram equation')
lbl_hsv_hist_txt.grid(row=0, column=3)

lbl_linear_contrast_txt = tk.Label(text='linear contrast')
lbl_linear_contrast_txt.grid(row=0, column=4)

open_filter_button = tk.Button(text="Open Image for low pass filter", command=open_filter_image)
open_filter_button.grid(row=1, column=0, columnspan=2)

open_contrast_button = tk.Button(text="Open Image for contrast", command=open_contrast_image)
open_contrast_button.grid(row=1, column=2, columnspan=3)

lbl_filter_usual = tk.Label()
lbl_filter_usual.grid(row=2, column=0)
lbl_filter_LoG = tk.Label()
lbl_filter_LoG.grid(row=2, column=1)

lbl_color_hist = tk.Label()
lbl_color_hist.grid(row=2, column=2)

lbl_hsv_hist = tk.Label()
lbl_hsv_hist.grid(row=2, column=3)

lbl_linear_contrast = tk.Label()
lbl_linear_contrast.grid(row=2, column=4)

lbl_filter_orig_txt = tk.Label(text='original image for filter')
lbl_filter_orig_txt.grid(row=3, column=0, columnspan=2)
lbl_contrast_orig_txt = tk.Label(text='original image for contrast')
lbl_contrast_orig_txt.grid(row=3, column=2, columnspan=3)

lbl_filter_orig = tk.Label()
lbl_filter_orig.grid(row=4, column=0, columnspan=2)
lbl_contrast_orig = tk.Label()
lbl_contrast_orig.grid(row=4, column=2, columnspan=3)

root.mainloop()
