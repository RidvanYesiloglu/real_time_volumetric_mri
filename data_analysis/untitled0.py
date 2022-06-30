# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:38:08 2022

@author: ridva
"""
import matplotlib.pyplot as plt

class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()

fig1, ax1 = plt.subplots()
#ax.set_aspect('equal') #<-not necessary 
ax1.set_ylim(0,3)
x = [0,1,2,3]
y = [1,1,2,2]

# plot a line, with 'linewidth' in (y-)data coordinates.       
l = data_linewidth_plot(x, y, ax=ax1, label='some 1 data unit wide line', 
                        linewidth=1, alpha=0.4)

plt.legend() # <- legend possible
plt.show()