import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageDraw
import pickle
from CustomNN import *

class DrawingApp(tk.Tk):
    def __init__(self, model: NeuralNetwork):
        super().__init__()
        self.model = model
        self.title('Pixel Drawing App')
        self.geometry('500x500')
        
        self.canvas_size = 280
        self.pixel_size = 10
        self.grid_size = self.canvas_size // self.pixel_size

        font = tkFont.Font(family="Arial", size=20, weight='bold')
        self.predict_box = tk.Label(self, height=1, width=20, font=font, text='Predicted: N/A')
        self.predict_box.pack()
        
        self.canvas = tk.Canvas(self, bg='white', width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(pady=20)
        
        self.clear_button = tk.Button(self, text='Clear', command=self.clear_canvas)
        self.clear_button.pack()
        
        self.save_button = tk.Button(self, text='Predict', command=self.predict)
        self.save_button.pack()
        
        self.canvas.bind('<B1-Motion>', self.draw)
        
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw_image = ImageDraw.Draw(self.image)
    
    def draw(self, event):
        x, y = (event.x // self.pixel_size) * self.pixel_size, (event.y // self.pixel_size) * self.pixel_size
        self.canvas.create_rectangle(x, y, x + self.pixel_size, y + self.pixel_size, fill='black', outline='black')
        self.draw_image.rectangle([x, y, x + self.pixel_size, y + self.pixel_size], fill='black')
        self.predict()
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw_image = ImageDraw.Draw(self.image)
    
    def predict(self):
        image = self.image.convert('L')
        image_array = np.array(image)
        
        binary_image = image_array < 127
        
        coords = np.column_stack(np.where(binary_image))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped = cropped.resize((28, 28))
        img = np.array(cropped).reshape(1, 784)
        
        y_pred = self.model.forward(img)
        predicted = y_pred.argmax()
        # print(f'Predicted: {predicted}')
        self.predict_box.config(text=f'Predicted: {predicted}')

        

if __name__ == '__main__':
    model = pickle.load(open('model.pk', 'rb'))
    App = DrawingApp(model=model)
    App.mainloop()
