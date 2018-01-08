import tkinter as tk                # python 3
from tkinter import font  as tkfont # python 3
from tkinter import *
from tkinter import filedialog as fd
import cv2
from random import shuffle
from tqdm import tqdm
import numpy as np
import os
import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

TRAIN_DIR = 'D:/train'
IMG_SIZE = 60
LR = 1e-3
TRAIN_DATA_NAME = 'dogvscatsTrainingData.npy'

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.geometry("640x640+100+50")
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)


        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        label = tk.Label(self, text="Cats Or Dog", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Tek Resim İle Test Et",
                            command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="Test Data ile Test Et",
                            command=lambda: controller.show_frame("PageTwo"))
        button1.pack()
        button2.pack()



class PageOne(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        def filepathfind():
            filename = fd.askopenfilename()
            theLabel3.config(text=filename)
            return filename

        def queryImageShow():
            path =theLabel3.cget("text")
            cv2.imshow("Image File", cv2.resize(cv2.imread(path, 0), (300, 300)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def predict():
            convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = fully_connected(convnet, 1024, activation='relu')
            convnet = dropout(convnet, 0.8)
            convnet = fully_connected(convnet, 2, activation='softmax')
            convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                                 name='targets')
            model = tflearn.DNN(convnet, tensorboard_verbose=0)
            model.load("my_cnn.tflearn")
            path = theLabel3.cget("text")
            image2D = np.array(cv2.resize(cv2.imread(path, 0), (IMG_SIZE, IMG_SIZE)))
            imageData = image2D.reshape(IMG_SIZE, IMG_SIZE, 1)
            prediction = model.predict([imageData])[0]
            sonuc = 'Cat: ', prediction[0], '\tDog: ', prediction[1]
            theLabelSonuc1.config(text=sonuc)
            theLabelSonuc1.pack()


        theLabel6 = Label(self, text="--------------------------------------------------------------------------")
        theLabel6.pack()
        theLabel6 = Label(self, text="")
        theLabel6.pack()

        theLabel2 = Label(self, text="Tek resim ile tahmin etme")
        theLabel2.pack()

        theLabelPath = Label(self, text="Resim Url :")
        theLabelPath.pack()

        theLabel3 = Label(self, text="")
        theLabel3.pack()

        fileButton1 = Button(self, text="Resim Bul")
        fileButton1.config(command=filepathfind)
        fileButton1.pack()

        tahminButton1 = Button(self, text="Tahmin Et")
        tahminButton1.config(command=predict)
        tahminButton1.pack()

        theLabelSonuc = Label(self, text="Sonuç : ")
        theLabelSonuc.pack()

        theLabelSonuc1 = Label(self, text="")
        theLabelSonuc1.pack()

        showButton = Button(self, text="Resme Bak")
        showButton.config(command=queryImageShow)
        showButton.pack()

        theLabel6 = Label(self, text="--------------------------------------------------------")
        theLabel6.pack()
        theLabel6 = Label(self, text="")
        theLabel6.pack()
        button = tk.Button(self, text="Anasayfa'ya dön", command=lambda: controller.show_frame("StartPage"))
        button.pack()



class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        def process_test_data(TEST_DIR):
            testing_data = []
            for img in tqdm(os.listdir(TEST_DIR)):
                path = os.path.join(TEST_DIR, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                testing_data.append([np.array(img), img_num])

            shuffle(testing_data)
            np.save('test_data44.npy', testing_data)
            return testing_data

        def findPathData():
            filename = fd.askdirectory(initialdir='.')
            theLabelData.config(text=filename)
            return filename

        def predictData():
            convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = max_pool_2d(convnet, 2)
            convnet = conv_2d(convnet, 32, 3, activation='relu')
            convnet = fully_connected(convnet, 1024, activation='relu')
            convnet = dropout(convnet, 0.8)
            convnet = fully_connected(convnet, 2, activation='softmax')
            convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',name='targets')
            model = tflearn.DNN(convnet, tensorboard_verbose=0)
            model.load("my_cnn.tflearn")
            Test_dir = theLabelData.cget("text")
            test_data = process_test_data(Test_dir)
            fig = plt.figure()
            cat = 'cat'
            dog = 'dog'
            value = 0.5
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for num, data in enumerate(test_data[:100]):
                img_num = data[1]
                img_data = data[0]
                y = fig.add_subplot(10, 10, num + 1)
                orig = img_data
                data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
                model_out = model.predict([data])[0]
                print(model_out)
                if np.argmax(model_out) == 1:
                    str_label = 'Dog'
                else:
                    str_label = 'Cat'
                y.imshow(orig, cmap='gray')
                plt.title(str_label)
                y.axes.get_xaxis().set_visible(False)
                y.axes.get_yaxis().set_visible(False)
                if img_num == cat:
                    if model_out[1] > value:
                        fn = fn + 1
                    else:
                        tn = tn + 1
                elif img_num == dog:
                    if model_out[1] > value:
                        tp = tp + 1
                    else:
                        fp = fp + 1

            totaltrue = tp + tn
            totalfalse = fn + fp
            total = totaltrue + totalfalse
            accuracy = (totaltrue / total) * 100
            err = (totalfalse / total) * 100
            tp1 = (tp / total) * 100
            tn1 = (tn / total) * 100
            fp1 = (fp / total) * 100
            fn1 = (fn / total) * 100
            theLabel8.config(text="Accuracy : %s " % accuracy)
            theLabel8.pack()
            theLabel9.config(text="Error Rate : %s " % err)
            theLabel9.pack()
            theLabel10.config(text="True Negative Rate : %s " % tn1)
            theLabel10.pack()
            theLabel11.config(text="True Positive Rate : %s " % tp1)
            theLabel11.pack()
            theLabel12.config(text="False Negative Rate : %s " % fn1)
            theLabel12.pack()
            theLabel13.config(text="False Positive Rate : %s " % fp1)
            theLabel13.pack()
            theLabel13.config(text="Confussion Matrix : ")
            theLabel13.pack()
            theLabel16.config(text="                                                                     TAHMİN                 ")
            theLabel16.pack()
            theLabel17.config(text="                                                    |     dog      |    cat   | ")
            theLabel14.config(text="GERCEK              | dog |           %s " % tn + "            " + "%s" % fp)
            theLabel14.pack()
            theLabel15.config(text="                                 |cat|           %s " % fn + "            " + "%s" % tp)
            theLabel15.pack()


        label = tk.Label(self, text="Test Dosyası ile Test Etme", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)


        theLabel5 = Label(self, text="Dosya Sec :")
        theLabel5.pack()

        theLabelData = Label(self, text="")
        theLabelData.pack()

        fileButton2 = Button(self, text="Dosya Bul")
        fileButton2.config(command=findPathData)
        fileButton2.pack()

        tahminButton2 = Button(self, text="Tahmin Et")
        tahminButton2.config(command=predictData)
        tahminButton2.pack()

        theLabel6 = Label(self, text="---------------------------------------------------------------")
        theLabel6.pack()

        theLabel7 = Label(self, text="Sonuclar")
        theLabel7.pack()
        theLabel6 = Label(self, text="")
        theLabel6.pack()

        theLabel8 = Label(self, text="")
        theLabel8.pack()
        theLabel9 = Label(self, text="")
        theLabel9.pack()
        theLabel10 = Label(self, text="")
        theLabel10.pack()
        theLabel11 = Label(self, text="")
        theLabel11.pack()
        theLabel12 = Label(self, text="")
        theLabel12.pack()
        theLabel13 = Label(self, text="")
        theLabel13.pack()
        theLabel18 = Label(self, text="")
        theLabel18.pack()
        theLabel16 = Label(self, text="")
        theLabel16.pack()
        theLabel17 = Label(self, text="")
        theLabel17.pack()
        theLabel14 = Label(self, text="")
        theLabel14.pack()
        theLabel15 = Label(self, text="")
        theLabel15.pack()

        button = tk.Button(self, text="Anasayfa'ya dön", command=lambda: controller.show_frame("StartPage"))
        button.pack()


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()