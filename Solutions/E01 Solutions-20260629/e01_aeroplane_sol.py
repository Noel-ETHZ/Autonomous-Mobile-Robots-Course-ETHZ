#written by Stefan Leutenegger, TU Munich, 27.10.2021
import numpy as np
import math as m
import vedo
  
def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                      [ 0, m.cos(theta),-m.sin(theta)],
                      [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
      return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                        [ 0           , 1, 0           ],
                        [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
      return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                        [ m.sin(theta), m.cos(theta) , 0 ],
                        [ 0           , 0            , 1 ]])

def apply_transform(*objs, T):
    for obj in objs:
        obj.apply_transform(np.linalg.inv(obj.transform.matrix))
        obj.apply_transform(T)

class Viewer:
    
    def __init__(self):
        self.plot = vedo.Plotter(axes=3) #with frame E axis
        self.plot.camera.SetPosition([-2, 10, -5])
        self.plot.camera.SetViewUp([0.2, -1.0, 0.0])
        self.plot.camera.SetFocalPoint([0, 0, 0])
        
        #load a nice aeroplane
        self.aeroplane = vedo.load(vedo.dataurl+"cessna.vtk")
        
        #put a sensible bounding box aorund it
        self.box = vedo.Box([0,0,0], 6, 6, 6).wireframe()
        
        #since the object was defined with "robotics" standard z-up, let's remember to fix this later
        #through a separate "Object" frame.
        self.T_BO = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        #sliders
        self.R = 0.0
        self.P = 0.0
        self.Y = 0.0
        self.slider_roll = self.plot.add_slider(
            self.sliderR, -180, 180, value=0, pos=1, title="roll deg = 0", show_value=False
        )
        self.slider_pitch = self.plot.add_slider(
            self.sliderP, -90, 90, value=0, pos=2, title="pitch deg = 0", show_value=False
        )
        self.slider_yaw = self.plot.add_slider(
            self.sliderY, -180, 180, value=0, pos=3, title="yaw deg = 0", show_value=False
        )
        self._update_slider_titles()

        self.Btx = vedo.Arrow(start_pt=(0,0,0), end_pt=(1.5,0,0), c="red", shaft_radius=0.01, head_radius=0.05, head_length=0.2)
        self.Bty = vedo.Arrow(start_pt=(0,0,0), end_pt=(0,1.5,0), c="green", shaft_radius=0.01, head_radius=0.05, head_length=0.2)
        self.Btz = vedo.Arrow(start_pt=(0,0,0), end_pt=(0,0,1.5), c="blue", shaft_radius=0.01, head_radius=0.05, head_length=0.2)

        self.T_BC0 = np.array([[0,0,1,0],[1,0,0,2],[0,1,0,0],[0,0,0,1]])
        self.C0tx = self.Btx.clone()
        self.C0ty = self.Bty.clone()
        self.C0tz = self.Btz.clone()

        self.T_BC1 = np.array([[0,0,1,0],[1,0,0,-2],[0,1,0,0],[0,0,0,1]])
        self.C1tx = self.Btx.clone()
        self.C1ty = self.Bty.clone()
        self.C1tz = self.Btz.clone()

        self.render(0,0,0)
        self.plot.show(self.aeroplane, self.box,
                       self.Btx, self.Bty, self.Btz,
                       self.C0tx, self.C0ty, self.C0tz,
                       self.C1tx, self.C1ty, self.C1tz,
                       resetcam=False, interactive=1)
    
    def render(self, r, p, y):
        roll = r / 180 * m.pi
        yaw = y / 180 * m.pi
        pitch = p / 180 * m.pi
        
        T_EB = np.eye(4).astype(float)
        T_EB[:3, :3] = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        T = T_EB @ self.T_BO

        apply_transform(self.aeroplane, T=(T_EB @ self.T_BO))
        apply_transform(self.Btx, self.Bty, self.Btz, T=T_EB)
        apply_transform(self.C0tx, self.C0ty, self.C0tz, T=(T_EB @ self.T_BC0))
        apply_transform(self.C1tx, self.C1ty, self.C1tz, T=(T_EB @ self.T_BC1))
        

    def _update_slider_titles(self):
        self.slider_roll.GetRepresentation().SetTitleText(f"roll deg = {self.R:.0f}")
        self.slider_pitch.GetRepresentation().SetTitleText(f"pitch deg = {self.P:.0f}")
        self.slider_yaw.GetRepresentation().SetTitleText(f"yaw deg = {self.Y:.0f}")
               
    def sliderR(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        self.R = value
        self._update_slider_titles()
        self.render(self.R, self.P, self.Y)
    def sliderP(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        self.P = value
        self._update_slider_titles()
        self.render(self.R, self.P, self.Y)
    def sliderY(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        self.Y = value
        self._update_slider_titles()
        self.render(self.R, self.P, self.Y)
    

if __name__ == "__main__":
    #start the interactive viewer
    viewer = Viewer()
