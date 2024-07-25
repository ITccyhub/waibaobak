import tkinter as tk
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
from threading import Thread
import sys
import torch.hub
import torch.nn as nn
from pathlib import Path

# 导入YOLOv5所需的库和模型文件
model_path = 'yolov5s.pt'  # 确保已经下载了YOLOv5s模型文件并提供了正确的路径
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

specific_roi_x1, specific_roi_y1 = 150, 150  # 特定区域的左上角坐标
specific_roi_x2, specific_roi_y2 = 250, 250  # 特定区域的右下角坐标
warning_threshold = 10  # 设定警告阈值，可以根据需要调整

is_detection_enabled = False  # 添加一个状态变量来控制是否进行目标检测

def label_mouse_callback(event, canvas):
    global specific_roi_x1, specific_roi_y1, specific_roi_x2, specific_roi_y2
    print("Clicked at ({}, {})".format(event.x, event.y))
    if event.type == cv2.EVENT_LBUTTONDOWN:
        specific_roi_x1, specific_roi_y1 = event.x, event.y
    elif event.type == cv2.EVENT_LBUTTONUP:
        specific_roi_x2, specific_roi_y2 = event.x, event.y
        cv2.rectangle(frame2_with_roi, (specific_roi_x1, specific_roi_y1), (specific_roi_x2, specific_roi_y2), (0, 0, 255), 2)
        image = cv2.cvtColor(frame2_with_roi, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=image)
        canvas.image = image

def calculate_distance(center_x, center_y, rect_x1, rect_y1, rect_x2, rect_y2):
    # 计算物体中心点到红色矩形框的最近边的距离
    rect_width = rect_x2 - rect_x1
    rect_height = rect_y2 - rect_y1
    dx = max(abs(center_x - rect_x1), abs(center_x - rect_x2))
    dy = max(abs(center_y - rect_y1), abs(center_y - rect_y2))
    distance = np.sqrt(dx * dx + dy * dy)
    return distance

def video():
    global frame2_with_roi, is_detection_enabled
    is_maximized = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    ret, frame1 = cap.read()
    if not ret:
        print("无法接收帧（流可能已结束）")
        exit()

    frame2_with_roi = frame1.copy()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    warning_var = tk.StringVar()
    warning_label = tk.Label(window, textvariable=warning_var, font=('Arial', 16), fg='red')
    warning_label.pack(side='bottom')

    def update_image():
        global frame2_with_roi, is_detection_enabled
        ret, frame2 = cap.read()
        if not ret:
            return

        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_with_roi = frame2.copy()

        if specific_roi_x1 is not None and specific_roi_y1 is not None and specific_roi_x2 is not None and specific_roi_y2 is not None:
            cv2.rectangle(frame2_with_roi, (specific_roi_x1, specific_roi_y1), (specific_roi_x2, specific_roi_y2), (0, 255, 0), 2)

        if is_detection_enabled:  # 只有当检测启用时才执行目标检测算法
            # 应用YOLOv5目标检测算法
            results = model(frame2)
            detections = results.xyxy[0].cpu().numpy()

            for (x, y, w, h, conf, cls) in detections:
                cv2.rectangle(frame2_with_roi, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                distance = calculate_distance(center_x, center_y, specific_roi_x1, specific_roi_y1, specific_roi_x2, specific_roi_y2)
                if distance < warning_threshold:
                    warning_var.set("警告：物体靠近！")
                else:
                    warning_var.set("")

                # 添加类别名称和置信度文本
                text = "{}: {:.2f}".format(results.names[int(cls)], conf)
                cv2.putText(frame2_with_roi, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        image = cv2.cvtColor(frame2_with_roi, cv2.COLOR_BGR2RGB)
        if is_maximized:
            image = cv2.resize(image, (canvas.winfo_width(), canvas.winfo_height()))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=image)
        canvas.image = image

        window.after(10, update_image)  # 每10毫秒更新一次画面
        print("Updating image")

    update_image()  # 开始更新画面

    window.mainloop()

def start_video():
    global is_detection_enabled
    is_detection_enabled = False  # 初始状态下不进行目标检测
    window.after(0, video)  # 将video函数放入主线程的事件队列中

def start_object_detection():
    global is_detection_enabled
    is_detection_enabled = True  # 启用目标检测
    print("Object detection started")

def enable_detection():
    global is_detection_enabled
    is_detection_enabled = True  # 启用目标检测
    print("Detection enabled")

window = tk.Tk()
window.bind('<Escape>', lambda event: window.quit())

# 创建顶部区域用于显示摄像头实时画面
top_frame = tk.Frame(window)
top_frame.pack(side='top', fill='both', expand=True)

# 创建警告标签并将其放置在顶部区域
warning_var = tk.StringVar()
warning_label = tk.Label(top_frame, textvariable=warning_var, font=('Arial', 16), fg='red')
warning_label.pack(side='bottom')

canvas = tk.Canvas(top_frame, width=640, height=320)  # 设置画布大小为摄像头画面的一半
canvas.pack(fill='both', expand=True)
canvas.bind("<Button-1>", lambda event: label_mouse_callback(event, canvas))
canvas.bind("<ButtonRelease-1>", lambda event: label_mouse_callback(event, canvas))

# 创建底部区域放置按钮
bottom_frame = tk.Frame(window)
bottom_frame.pack(side='bottom', fill='x')
b1 = tk.Button(bottom_frame, text='打开摄像头', width=15, height=2, command=start_video)
b1.pack(side='left')
b2 = tk.Button(bottom_frame, text='开始', width=15, height=2, command=start_object_detection)  # 点击后启动人脸检测
b2.pack(side='left')
b3 = tk.Button(bottom_frame, text='结束', width=15, height=2, command=exit)  # 这里可以添加停止人脸检测的逻辑
b3.pack(side='left')

window.title('报警系统')
window.geometry('640x480')
window.mainloop()
