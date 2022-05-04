import time
import tkinter as tk
from PIL import ImageTk, Image


class Overlay:
    def __init__(self, gameOverlayState):
        title_font = ("Helvetica", 20)
        font = ("Helvetica", 14)
        self.root = tk.Tk()
        self.root.geometry('1000x800+100+100')

        image1 = Image.open('./instruction.png')
        test = ImageTk.PhotoImage(image1)
        label1 = tk.Label(image=test)
        label1.image = test
        label1.place(x=0, y=0)

    def show(self, gameOverlayState):
        self.root.update()

        while True:
            state = ""
            if not gameOverlayState.empty():
                state = gameOverlayState.get()

            if state == "help off":
                self.root.destroy()
                break
            else:
                self.root.update()
