from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.base import EventLoop
from kivy.config import Config
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import time

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

#MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print ('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
else:
    pass


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def flames(str1,str2,str3,str4):
    text="datasets/videos/"
    str1=text+str1+'.mp4'
    str2=text+str2+'.mp4'

    str3=text+str3+'.mp4'
    str4=text+str4+'.mp4'

    print (str1,str2,str3,str4)

    
    count1=0
    count2=0
    count3=0
    count4=0
    h=0
    f1=0
    f2=0
    f3=0
    f4=0
    q=0
    import cv2
    cap1 = cv2.VideoCapture(str1)
    cap2 = cv2.VideoCapture(str2)
    cap3 = cv2.VideoCapture(str3)
    cap4 = cv2.VideoCapture(str4)
    top= cv2.imread('top.png')
    right= cv2.imread('right.png')
    left= cv2.imread('left.png')
    bottom= cv2.imread('bottom.png')

    top = cv2.resize(top, (300,300))
    left = cv2.resize(left, (300,300))
    right = cv2.resize(right, (300,300))
    bottom = cv2.resize(bottom, (300,300))

    s=top

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
       ret = True
       while ret: 
          ret,image_np1 = cap1.read()
          image_np1=cv2.resize(image_np1,(300,300))
          image_np1_expanded = np.expand_dims(image_np1, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np1_expanded})

          vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np1,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2,
                        )

          final_score = np.squeeze(scores)    
          if h%10==0:
            for i in range(100):
              if scores is None or final_score[i] > 0.5:
                count1 = count1 + 1

          cv2.resize(image_np1,(1280,960))
          h+=1
          font = cv2.FONT_HERSHEY_SIMPLEX
          if count1 <10:
            cv2.putText(image_np1,str(count1),(30,50), font, 2,(255,255,255),2,cv2.LINE_AA)
          else:
            cv2.putText(image_np1,str(count1),(30,50), font, 2,(0,0,255),2,cv2.LINE_AA)
            f1=1

          ret,image_np2 = cap2.read()
          image_np2=cv2.resize(image_np2,(300,300))
          image_np2_expanded = np.expand_dims(image_np2, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np2_expanded})

          vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np2,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2,
                        )

          final_score = np.squeeze(scores)    
          if h%10==0:
            for i in range(100):
              if scores is None or final_score[i] > 0.5:
                count2 = count2 + 1

          cv2.resize(image_np2,(1280,960))
          font = cv2.FONT_HERSHEY_SIMPLEX
          if count2 <10:
            cv2.putText(image_np2,str(count2),(30,50), font, 2,(255,255,255),2,cv2.LINE_AA)
          else:
            cv2.putText(image_np2,str(count2),(30,50), font, 2,(0,0,255),2,cv2.LINE_AA)
            f2=1

          ret,image_np3 = cap3.read()
          image_np3 = cv2.resize(image_np3,(300,300))
          image_np3_expanded = np.expand_dims(image_np3, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np3_expanded})

          vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np3,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2,
                        )

          final_score = np.squeeze(scores)    
          if h%10==0:
            for i in range(100):
              if scores is None or final_score[i] > 0.5:
                count3 = count3 + 1
          cv2.resize(image_np3,(1280,960))
          font = cv2.FONT_HERSHEY_SIMPLEX
          if count3 <10:
            cv2.putText(image_np3,str(count3),(30,50), font, 2,(255,255,255),2,cv2.LINE_AA)
          else:
            cv2.putText(image_np3,str(count3),(30,50), font, 2,(0,0,255),2,cv2.LINE_AA)
            f3=1


          ret,image_np4 = cap4.read()
          image_np4=cv2.resize(image_np4,(300,300))
          image_np4_expanded = np.expand_dims(image_np4, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np4_expanded})

          vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np4,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2,
                        )

          final_score = np.squeeze(scores)    
          if h%10==0:
            for i in range(100):
              if scores is None or final_score[i] > 0.5:
                count4 = count4 + 1
          cv2.resize(image_np4,(1280,960))

          font = cv2.FONT_HERSHEY_SIMPLEX
          if count4 <10:
            cv2.putText(image_np4,str(count4),(30,50), font, 2,(255,255,255),2,cv2.LINE_AA)
          else:
            cv2.putText(image_np4,str(count4),(30,50), font, 2,(0,0,255),2,cv2.LINE_AA)
            f4=1

          height,width,_=image_np1.shape
          img = np.ones((height,width,3), np.uint8)

          numpy_vertical = np.concatenate((img, image_np2), axis=0)
          numpy_vertical_concat_1 = np.concatenate((numpy_vertical, img), axis=0)
          temp=max(count1,count2,count3,count4)
          q+=1
          # if q>10:
          #     q=0
          #     if f1==1:
          #         f1=0
          #         s=top
          #         count1=0
          #     elif f2==1:
          #         f2=0
          #         s=left
          #         count2=0
          #     elif f3==1:
          #         f3=0
          #         s=right
          #         count3=0
          #     elif f4==1:
          #         f4=0
          #         s=bottom
          #         count4=0
          if q>10:
              q=0
              if temp==count1 and f1==1:
                count1=0
                s=top
                f1=0
              if temp==count2 and f2==1:
                count2=0
                s=left
                f2=0
              if temp==count3 and f3==1:
                count3=0
                s=right
                f3=0
              if temp==count4 and f4==1:
                count4=0
                s=bottom
                f4=0

          numpy_vertical2 = np.concatenate((s, image_np4), axis=0) 
          numpy_vertical_concat_2 = np.concatenate((image_np1,numpy_vertical2), axis=0)

          numpy_vertical = np.concatenate((img, image_np3), axis=0)
          numpy_vertical_concat_3 = np.concatenate((numpy_vertical, img), axis=0)

          final1 = np.concatenate((numpy_vertical_concat_1, numpy_vertical_concat_2), axis=1)
          final = np.concatenate((final1, numpy_vertical_concat_3), axis=1)
          final= cv2.resize(final,(1380,700))

          cv2.imshow('N', final)
          cv2.waitKey(10)

class LoginScreen(GridLayout):
    def on_start(self):
        Logger.info('This is a Simple boredom app which gives Your most probable relationsip')

    def __init__(self):
        super(LoginScreen, self).__init__()
        EventLoop.ensure_window()
        EventLoop.window.title = self.title = 'Rockivy | Kivy App Contest 2014'
        self.rows=4
        self.cols=3

        lbl1=Label(text=" ",italic=True, bold=True)
        lbl2=Label(text=" ",italic=True, bold=True)
        lbl3=Label(text=" Traffic signal ",italic=True, bold=True)
        lbl4=Label(text=" ",italic=True, bold=True)
        lbl5=Label(text=" ",italic=True, bold=True)
        lbl6=Label(text=" ",italic=True, bold=True)
        txt1=TextInput(multiline=False, font_size=20)
        txt2=TextInput(multiline=False, font_size=20)
        txt3=TextInput(multiline=False, font_size=20)
        txt4=TextInput(multiline=False, font_size=20)
        btn1=Button(text="Exit",italic=True)
        btn1.bind(on_press=lambda *a:App.get_running_app().stop())
        btn2=Button(text="OK",italic=True)
        # print txt2.text
        if not txt1=='' or txt2=='' or txt3=='' or txt4=='':
            btn2.bind(on_press=lambda *a:flames(txt1.text,txt2.text,txt3.text,txt4.text))
        self.add_widget(lbl1)
        self.add_widget(txt1)
        self.add_widget(lbl2)
        self.add_widget(txt2)
        self.add_widget(lbl3)
        self.add_widget(txt3)
        self.add_widget(lbl4)
        self.add_widget(txt4)
        self.add_widget(lbl5)
        self.add_widget(btn1)
        self.add_widget(lbl6)
        self.add_widget(btn2)


class Adaptive_Traffic_Light_control(App):
    def build(self):
        l = Label(text='Hello world')
        return LoginScreen()

if __name__ == "__main__":
    Config.set('graphics', 'width', '1380')
    Config.set('graphics', 'height', '700')  # 16:9
    Config.set('graphics', 'resizable', '0')
    Adaptive_Traffic_Light_control().run() 