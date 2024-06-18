import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import json
root = tk.Tk()
frame = tk.Frame()
frame.pack()

global img
global image
global rect

def write_coordinate_json():    
    # Data to be written
    global img_path

    fn = img_path.split("/")
    print(fn[-1])

    dictionary ={
        'file_name':fn[-1],
        'lt':(x1,y1),
        'br':(x2,y2)
    }
  
    with open("coordinate.json", "w") as outfile:
        json.dump(dictionary, outfile)

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def oas():
    global img_path
    
    img_path = filedialog.askopenfilename(title='選擇',
                                        filetypes=[
                                            ('All Files','*'),
                                            ("jpeg files","*.jpg"),
                                            ("png files","*.png"),
                                            ("gif files","*.gif")])

    global canvas
    global img
    global rect
    

    i = Image.open(img_path)

    image_x,image_y = i.size

    img = ImageTk.PhotoImage(i)

    
    canvas.config(width=image_x,height=image_y)
    canvas.create_image(0,0,anchor='nw',image=img)
    rect = canvas.create_rectangle(0,0,0,0,outline='red',width=5)

def push(event):
    global x1,y1
    x1,y1=event.x,event.y

def motion(event):
    global x1,y1
    global x2,y2
    global rect
    global canvas
    x2,y2 =event.x,event.y

    canvas.coords(rect,x1,y1,x2,y2)

def Release(event):
    global x1,y1
    global x2,y2
    global rect
    global canvas
    x2,y2 =event.x,event.y
    canvas.coords(rect,x1,y1,x2,y2)

def getcoord():
    mylabel.insert(tk.END,  '(' + str(x1) + ', '+str(y1)+')、(' + str(x2) + ', '+str(y2)+')')
    mylabel.insert(tk.END,  '\n')
    write_coordinate_json()
    # mylabel.configure(text = '(' + str(x1) + ', '+str(y1)+')、(' + str(x2) + ', '+str(y2)+')')

x1=0
x2=0
y1=0
y2=0

canvas = tk.Canvas(frame,width=100,height=100)
canvas.pack()

canvas.bind('<Button-1>',push) # 滑鼠左鍵按下
canvas.bind('<B1-Motion>',motion) # 滑鼠左鍵按下並移動
canvas.bind('<ButtonRelease-1>',Release) # 滑鼠左鍵釋放

tk.Button(root, text="開啟圖片",command = oas).pack()
tk.Button(root, text="取得座標",command = getcoord).pack()
# mylabel = tk.Label(root, text='')
mylabel = tk.Text(root, height = 5, width = 25)
mylabel.pack()
root.mainloop()