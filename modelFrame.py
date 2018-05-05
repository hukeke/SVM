# _*_ coding:utf-8 _*_

import tkinter
from tkinter import filedialog
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import *


def opentrain():
    train = filedialog.askopenfilename(title='打开train', filetypes=[('train', '*.xls'), ('All Files', '*')])
    return train

def opentest():
    test = filedialog.askopenfilename(title='打开test', filetypes=[('test', '*.xls'), ('All Files', '*')])
    return test


def modelPls():
    train = opentrain
    test = opentest
    error = 0
    return error


def modelSvm():
    return 1


def modelselect(*args):
    if (numberChosen.get()=="PLS"):
        print(modelPls())
    if (numberChosen.get()=="SVM"):
        print(modelSvm())


def show():
    t.insert(END, modelPls())


root = tkinter.Tk()

root.title("建模预测")

frame = Frame(root)
frame.grid()

ttk.Label(root, text="选择一个模型", font=('宋体', 12, 'bold'),).grid(column=1, row=0)
ttk.Label(root, text="均方误差", font=('宋体', 12, 'bold'),).grid(column=1, row=2)

t = tkinter.Text(root, width=15, height=3)
t.grid(column=1, row=3)

btn1 = tkinter.Button(root, text='打开train', font=('宋体', 15, 'bold'), command=opentrain).grid(column=2, row=1)
btn2 = tkinter.Button(root, text='打开test', font=('宋体', 15, 'bold'), command=opentest).grid(column=3, row=1)

number = tk.StringVar()
numberChosen = ttk.Combobox(root, textvariable=number)
numberChosen['values'] = ('PLS', 'SVM')     # 设置下拉列表的值
numberChosen['values'] = ('PLS', 'SVM')     # 设置下拉列表的值

numberChosen.current(0)    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
numberChosen.bind("<<ComboboxSelected>>", modelselect)
numberChosen.grid(column=1, row=1)

btn3 = tkinter.Button(root, text='查看均方误差', font=('宋体', 15, 'bold'), command=show)
btn3.grid(column=4, row=1)

root.mainloop()







