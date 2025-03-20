import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import tkinter as tk
from tkinter import Canvas, Button, Entry, Label
from PIL import Image, ImageGrab

# Загружаем обученную модель MNIST
model = keras.models.load_model('mnist_model.h5')  # Убедись, что файл существует

# Создаем окно
window = tk.Tk()
window.title("Распознавание цифр")

# Создаем холст для рисования
canvas = Canvas(window, width=280, height=280, bg="white")
canvas.pack()

# Функции для рисования
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x + 10, y + 10, fill="black", outline="black")  

def switch_focus(event):
    window.event_generate("<Tab>")  

label = Label(window, text="Введите текст:")
label.pack(pady=5)

entry = Entry(window)
entry.pack(pady=5)

button = Button(window, text="Распознать цифру")
button.pack(pady=5)

window.bind("<Alt-Tab>", switch_focus)

window.mainloop()
