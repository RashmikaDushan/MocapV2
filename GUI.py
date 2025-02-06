import tkinter as tk
from tkinter import Scrollbar,Button,Label,Text,Checkbutton,Entry,Scale
import cv2 as cv
from PIL import Image, ImageTk 

camera_count = 6
camerasettingslog = "Camera Settings Log"
cameranumber = 0
applyforall = False
exposure = 0
contrast = 0

root = tk.Tk(screenName=None, baseName="Mocap", className="Mocap", useTk=1)
root.title("Mocap")
root.geometry("1200x800")
root.config(bg="skyblue")
root.bind('<Escape>', lambda e: root.quit())

frameCamFeed = tk.Frame(root,borderwidth=1,bg="black")
frameCamFeed.pack(fill="both", side=tk.TOP, expand='True')

def main_thread():
    print("Main Thread")


###############################################################
frameBottom = tk.Frame(root,bg="green")
frameBottom.pack(fill="both", side=tk.BOTTOM, expand='True')

# Camera Settings Frame
frameCamSettings = tk.Frame(frameBottom)
frameCamSettings.pack(fill="both", side=tk.LEFT, expand='True')

frameCamSettingsTitle = Label(frameCamSettings, text="Camera Settings")
frameCamSettingsTitle.config(font=("Arial", 14))
frameCamSettingsTitle.pack()

frameCamSettingsButtons = tk.Frame(frameCamSettings)
frameCamSettingsButtons.pack()
detectCamerasButton = Button(frameCamSettingsButtons, text="Detect Cameras", command=lambda: print("Detecting Cameras"))
detectCamerasButton.pack(side=tk.LEFT)
buttonCamerasOn = Button(frameCamSettingsButtons, text="Cameras On", command=lambda: print("Cameras On"))
buttonCamerasOn.pack(side=tk.LEFT)

textCamSettingsLog = Text(frameCamSettings,width=20,height=5)
textCamSettingsLog.pack(side=tk.TOP,fill="both", expand='True')

checkApplyForAll = Checkbutton(frameCamSettings, text="Apply for all cameras",variable=applyforall)
checkApplyForAll.pack(side=tk.TOP)

frameCamNumber = tk.Frame(frameCamSettings)
frameCamNumber.pack(side=tk.TOP)
labelCamNumber = Label(frameCamNumber, text="Camera Number")
labelCamNumber.pack(side=tk.LEFT)
inputCamNumber = Entry(frameCamNumber, textvariable=cameranumber)
inputCamNumber.pack(side=tk.RIGHT)

frameExposure = tk.Frame(frameCamSettings)
frameExposure.pack(side=tk.TOP)
labelExposure = Label(frameExposure, text="Exposure")
labelExposure.pack(side=tk.LEFT)
sliderExposure = Scale(frameExposure, 
           from_ = -100, to = 100,  
           orient = tk.HORIZONTAL,variable=exposure)
sliderExposure.pack(side=tk.RIGHT)

frameContrast = tk.Frame(frameCamSettings)
frameContrast.pack(side=tk.TOP)
labelContrast = Label(frameContrast, text="Contrast")
labelContrast.pack(side=tk.LEFT)
sliderContrast = Scale(frameContrast, 
           from_ = -100, to = 100,  
           orient = tk.HORIZONTAL,variable=contrast)
sliderContrast.pack(side=tk.RIGHT)

# Camera Filters Frame
frameCamFilters = tk.Frame(frameBottom,bg="blue")
frameCamFilters.pack(fill="both", side=tk.LEFT, expand='True')

frameIntrinsics = tk.Frame(frameBottom,bg="yellow")
frameIntrinsics.pack(fill="both", side=tk.LEFT, expand='True')

frameExtrinsics = tk.Frame(frameBottom,bg="purple")
frameExtrinsics.pack(fill="both", side=tk.LEFT, expand='True')

frameLocate = tk.Frame(frameBottom,bg="orange")
frameLocate.pack(fill="both", side=tk.LEFT, expand='True')

# label_widget = Label(root) 
# label_widget.pack() 
# label = tk.Label(root, text="Mocap V2")
# # button1 = Button(root, text="Open Camera", command=open_camera(frame=images[0][0], label_widget=label_widget)) 
# # button1.pack()
# h = Scrollbar(root, orient='horizontal')
# h.config(command=t.xview)
root.mainloop()