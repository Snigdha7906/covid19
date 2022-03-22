
import streamlit as st
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image 
import time
import copy
import imutils
from scipy.spatial import distance as dist



neural_net = cv2.dnn.readNet("yolov4-tiny-obj_final.weights", "yolov4-tiny-obj.cfg")


classes = ["without_mask", "with_mask"]   #Initialize an array to store output labels 
names_of_layer = neural_net.getLayerNames() 
#Store the names of model’s layers obtained using getLayerNames() of OpenCV
output_layers = [names_of_layer[i-1] for i in neural_net.getUnconnectedOutLayers()]
    

    
def obj_detection(my_img, cv,MIN_DIST):
    


    #getUnnnectedOutLayers() returns indexes of layers with unconnected output
    colors = [(255,0,0),(0,255,0)]
    #RGB values selected randomly from 0 to 255 using np.random.uniform()
    # Image loading

    if cv==0:
        newimg = np.array(my_img.convert('RGB')) #Convert the image into RGB  
        img = cv2.cvtColor(newimg,1) #cvtColor()
    # #Store the height, width and number of color channels of the image        
    # img = cv2.resize(my_img, None, fx=0.8, fy=0.8)
    else:
        img=my_img
    
    height,width,channels = img.shape  
    

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), False, crop=False)
    neural_net.setInput(blob)
    outs = neural_net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []
    centroids=[]
    results=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
               # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

 # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                centroids.append((center_x, center_y))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = [(255,0,0),(0,255,0)]
    font = cv2.FONT_HERSHEY_PLAIN


    if len(indexes) > 0:
       for i in indexes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    violate=set()

    if len(results) >=2:
        centr=np.array([r[2] for r in results])
        d=dist.cdist(centr, centr, metric="euclidean")
        for i in range(0, d.shape[0]):
            for j in range(i+1, d.shape[1]):
                if(d[i,j]< MIN_DIST):
                    violate.add(i)
                    violate.add(j)
        for (i, (wgh, coor, c)) in enumerate(results):
            (x,y,w,h)= coor
            (cx,cy)=c
            t="No"
            col=(0,255,0)
            if i in violate:
                t="Yes"
                col=(255,0,0)
            cv2.putText(img, t, (x,y-10),font,0.85, col, 2)

    for i in range(len(boxes)):
       if i in indexes:
            x, y, w, h = boxes[i]
            col=(0,255,0)
            if class_ids[i]==0:
                col=(255,0,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
    cv2.putText(img, "Social Distancing Violations: {}".format(len(violate)), (10, img.shape[0] - 25), font, 0.85, (0, 0, 255), 3)
    return img

def main():    
    st.title("Welcome to COVID19 PROTOCOLS DETECTOR") #Title displayed in UI using streamlit.title()
    #Display some text on UI using streamlit.write()
    st.write("You can view real-time whether person is wearing Mask and violating Social Distancing. Select one of the following options to proceed:")
    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))
     #streamlit.radio() inserts a radio button widget 

    #If user selects 2nd option:
    if choice == "Choose an image of your choice":
        image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])

        if image_file is not None:
            my_img = Image.open(image_file)
        else:
            my_img= Image.open("crowd.jpg")
            
    elif choice == "See an illustration":
        #display the example image
        my_img = Image.open("crowd.jpg")
    # MIN_DIST=150

    MIN_DIST= st.sidebar.slider("Min Distance", 50,200,100,10)
    img=obj_detection(my_img,0,MIN_DIST)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    column1, column2 = st.columns(2)
    #Display subheading on top of input image 
    column1.subheader("Input image") #streamlit.subheader()
    st.text("") #streamlit.text() writes preformatted and fixed-width text
    #Display the input image using matplotlib
    plt.figure(figsize = (16,16)) 
    plt.imshow(my_img)
    column1.pyplot(use_column_width=True)
    st.text("") #preformatted and fixed-width text
    column2.subheader("Output image") #Title on top of the output image
    st.text("")
    #Plot the output image with detected objects using matplotlib
    plt.figure(figsize = (20,20))
        
    plt.imshow(img) #show the figure
    column2.pyplot(use_column_width=True) #actual plotting
    


    st.title("Webcam Live Detection")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=obj_detection(frame,1, MIN_DIST)
        FRAME_WINDOW.image(img)
    else:
        st.write('Stopped')
        

if __name__ == '__main__':
    main()