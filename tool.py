import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import math
from main import process_img, load_train_config
import skimage.io as ski_io

class terrainio:
    def __init__(self,master):
        self.master=master
        self.color_fg='white'
        self.color_bg='white'
        self.x=None
        self.y=None
        self.penwidth=5
        self.color_river = f'#668cff'
        self.color_basin = f'#33ffad'
        self.color_peaks = f'#994d00'
        self.color_ridge = f'#7575a3'
        self.enable_realtime_edid = False

        self.image_data = Image.new("RGB", (256,256), (255, 255, 255))
        self.image_data.save("./test.jpg")
        self.draw_data = ImageDraw.Draw(self.image_data)

        #temp
        self.river_data = Image.new("L", (256, 256), (255))
        self.basin_data = Image.new("L", (256, 256), (255))
        self.peaks_data = Image.new("L", (256, 256), (255))
        self.ridge_data = Image.new("L", (256, 256), (255)) 

        self.riverdraw_data = ImageDraw.Draw(self.river_data)
        self.basindraw_data = ImageDraw.Draw(self.basin_data)
        self.peaksdraw_data = ImageDraw.Draw(self.peaks_data)
        self.ridgedraw_data = ImageDraw.Draw(self.ridge_data)

        self.sketchmapimage = self.riverdraw_data
        self.draw_all = True

        self.rivervalue = 0.25
        self.basinvalue = 0.4
        self.peaksvalue = 0.7
        self.ridgevalue = 1
        self.model = load_train_config()
        self.running = False

        self.initiate_board()
        
    def paint(self, in_m):

        if self.x and self.y:
            self.draw.create_line(
                self.x, 
                self.y, 
                in_m.x, in_m.y, 
                width=self.penwidth,
                fill=self.color_fg,
                capstyle=ROUND,
                smooth=False)

            self.draw_data.line(
                [(self.x, 
                self.y), 
                (in_m.x, in_m.y)], 
                width=self.penwidth,
                fill=self.color_fg,)   

            #temp
            if self.draw_all:
                self.riverdraw_data.line(
                [(self.x, 
                self.y), 
                (in_m.x, in_m.y)], 
                width=self.penwidth,
                fill=(255),)

                self.basindraw_data.line(
                [(self.x, 
                self.y), 
                (in_m.x, in_m.y)], 
                width=self.penwidth,
                fill=(255),)

                self.peaksdraw_data.line(
                [(self.x, 
                self.y), 
                (in_m.x, in_m.y)], 
                width=self.penwidth,
                fill=(255),)

                self.ridgedraw_data.line(
                [(self.x, 
                self.y), 
                (in_m.x, in_m.y)], 
                width=self.penwidth,
                fill=(255),)

            else:
                self.sketchmapimage.line(
                [(self.x, 
                self.y), 
                (in_m.x, in_m.y)], 
                width=self.penwidth,
                fill=(0),)

             # coords = []
             # d = self.penwidth * 2
             # for i in d:
             #     xr = i - self.penwidth
             #     yr = self.penwidth - xr
             #     yr2 = -self.penwidth + xr
             #     disX = math.sqrt((self.x + xr)**2 - (in_m.x + xr)**2)
             #     disY1 = math.sqrt((self.y + yr)**2 - (in_m.y - yr)**2)
             #     disY2 = math.sqrt((self.y - yr)**2 - (in_m.y + yr)**2)
             #     coords.append(self.get_coordinates_on_line())

        self.x = in_m.x
        self.y = in_m.y

        self.image_data.save("./test.jpg")
        self.refresh_image()

    def get_coordinates_on_line(x0, y0, x1, y1):
        #impliment bresenham algorithm
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line

    def reset(self, _):
        self.x=None
        self.y=None
        if self.enable_realtime_edid:
            self.export_image_data_experimental()
        self.refresh_image()

    def clear(self):
        self.draw.delete(ALL)
        self.image_data = Image.new("RGB", (256,256), (255, 255, 255))
        self.draw_data = ImageDraw.Draw(self.image_data)
        self.image_data.save("./test.jpg")
        self.refresh_image()

        #temp
        self.river_data = Image.new("L", (256, 256), (255))
        self.basin_data = Image.new("L", (256, 256), (255))
        self.peaks_data = Image.new("L", (256, 256), (255))
        self.ridge_data = Image.new("L", (256, 256), (255)) 

        self.riverdraw_data = ImageDraw.Draw(self.river_data)
        self.basindraw_data = ImageDraw.Draw(self.basin_data)
        self.peaksdraw_data = ImageDraw.Draw(self.peaks_data)
        self.ridgedraw_data = ImageDraw.Draw(self.ridge_data)

    def change_element(self):
        return

    def change_size(self, e):
        self.penwidth = int(e)

    def show_legend(self):
        self.colorchoosing = Frame(self.master, padx = 5, pady = 5)
        Button(self.colorchoosing, text="river",font=('arial 11'), command=self.river).grid(row=0, column=0)
        Button(self.colorchoosing, text="basin",font=('arial 11'), command=self.basin).grid(row=1, column=0)
        Button(self.colorchoosing, text="peak",font=('arial 11'), command=self.peaks).grid(row=2, column=0)
        Button(self.colorchoosing, text="ridge",font=('arial 11'), command=self.ridge).grid(row=3, column=0)
        Button(self.colorchoosing, text="eraser",font=('arial 11'), command=self.eraser).grid(row=4, column=0)
        Scale(self.colorchoosing, 
              label="size", 
              font=('arial 11'), 
              command=self.change_size).grid(row=5, column=0)
        self.colorchoosing.pack(side=LEFT)

    def conv2rgb(self, hex):
        h = hex.lstrip('#')
        c = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return c

    def river(self):
        self.color_fg = self.color_river

        self.sketchmapimage = self.riverdraw_data
        self.draw_all = False

    def basin(self):
        self.color_fg = self.color_basin

        self.sketchmapimage = self.basindraw_data
        self.draw_all = False

    def peaks(self):
        self.color_fg = self.color_peaks

        self.sketchmapimage = self.peaksdraw_data
        self.draw_all = False

    def ridge(self):
        self.color_fg = self.color_ridge

        self.sketchmapimage = self.ridgedraw_data
        self.draw_all = False

    def eraser(self):
        self.color_fg = 'white'
        self.draw_all = True

    def refresh_image(self):
        self.img = Image.open("./IHopeThisWorks.png")
        self.tatras = ImageTk.PhotoImage(self.img)
        self.hm.delete(ALL)
        self.hm.create_image(128, 128, image=self.tatras)

    def export_image_data(self):
        if self.running:
            return
        self.draw.postscript(file="./tmp_canvas.eps",
                             colormode="color",
                             width=256,
                             height=256,
                             pagewidth=256-1,
                             pageheight=256-1)
        data = ski_io.imread("./tmp_canvas.eps")
        
        c_river = self.conv2rgb(self.color_river)
        c_basin = self.conv2rgb(self.color_basin)
        c_peaks = self.conv2rgb(self.color_peaks)
        c_ridge = self.conv2rgb(self.color_ridge)

        rivers = np.zeros(shape=[256, 256])
        basins = np.zeros(shape=[256, 256]) 
        peaks  = np.zeros(shape=[256, 256])
        ridges = np.zeros(shape=[256, 256])

        data2 = np.zeros(shape=[256, 256])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (data[i][j] == c_river).all():
                     rivers[i][j] = 1
                    # data2[i][j] = self.rivervalue
                elif (data[i][j] == c_basin).all():
                     basins[i][j] = 1
                    # data2[i][j] = self.basinvalue
                elif (data[i][j] == c_peaks).all():
                     peaks[i][j] = 1
                    # data2[i][j] = self.peaksvalue
                elif (data[i][j] == c_ridge).all():
                     ridges[i][j] = 1
                    # data2[i][j] = self.ridgevalue

        # plt.imsave("./Sketch.png", data2, cmap="Greys")

        # data2 = np.expand_dims(data2, axis=0)  
        # data2 = np.expand_dims(data2, axis=-1)  

        # rivers = np.expand_dims(rivers, axis=-1)
        rivers = np.expand_dims(rivers, axis=0)
        # basins = np.expand_dims(basins, axis=-1)
        basins = np.expand_dims(basins, axis=0)
        # peaks = np.expand_dims(peaks, axis=-1)
        peaks = np.expand_dims(peaks, axis=0)
        # ridges = np.expand_dims(ridges, axis=-1)
        ridges = np.expand_dims(ridges, axis=0)

        sketch_map = np.stack((ridges, rivers, peaks, basins), axis=3)
        print(sketch_map.shape)
        # sketch_map = np.squeeze(sketch_map, axis=0)

        # data2 = (data2 - .5) * 1.5
        self.running = True
        process_img(sketch_map,  self.model)
        self.running = False
        self.refresh_image()

    def export_image_data_experimental(self):

        rivers = 1 - np.array(self.river_data) / 255
        basins = 1 - np.array(self.basin_data) / 255
        peaks  = 1 - np.array(self.peaks_data) / 255
        ridges = 1 - np.array(self.ridge_data) / 255

        rivers = np.expand_dims(rivers, axis=0)
        basins = np.expand_dims(basins, axis=0)
        peaks = np.expand_dims(peaks, axis=0)
        ridges = np.expand_dims(ridges, axis=0)

        sketch_map = np.stack((ridges, rivers, peaks, basins), axis=3)
        
        process_img(sketch_map,  self.model)
        self.refresh_image()
        
    def switch_rt(self):
        self.enable_realtime_edid = not self.enable_realtime_edid

    def initiate_board(self):

        self.show_legend()
        self.draw = Canvas(app, width=256, height=256, bg = "white")
        self.draw.bind('<B1-Motion>', self.paint)
        self.draw.bind('<ButtonRelease-1>', self.reset)
        self.draw.pack(side=LEFT)

        self.hm = Canvas(app, width=256, height=256, bg = "grey")
        self.hm.pack(side=LEFT)
        self.img = Image.open("./IHopeThisWorks.png")
        self.tatras = ImageTk.PhotoImage(self.img)
        self.hm.create_image(128, 128, image=self.tatras)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)


        menu.add_cascade(label='options', menu=filemenu)
        filemenu.add_command(label='clear', command=self.clear)
        filemenu.add_command(label='convert', command=self.export_image_data)
        filemenu.add_command(label='real time', command=self.switch_rt)
        self.export_image_data()



if __name__ == '__main__':

    app = Tk()
    app.title('draw terrain')

    terrainio(app)

    app.mainloop()