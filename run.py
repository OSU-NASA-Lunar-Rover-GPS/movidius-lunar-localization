from tkinter import *

class FullScreenApp(object):

    def __init__(self, master, **kwargs):
        self.master=master
        self._geom='1920x1080+0+0'
        master.bind('<Escape>',self.end_fullscreen)
    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom

root=tk.Tk()
app=FullScreenApp(root)
root.mainloop()
