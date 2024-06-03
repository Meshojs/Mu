import easygui
import sklearn
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tkinter
from tkinter import *
import os
import cv2
import numpy as np
import datetime
import tkinter.messagebox


# made with love ;D
def load_img_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        label = filename.split(".")[0]
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)
    return images, labels


get_data = []


def GET():

    global get_data
    # l_one = e_label_one.get()
    f_one = e_folder_one.get()
    # l_two = e_label_two.get()
    f_two = e_folder_two.get()

    if f_one == "" or f_two == "":
        easygui.msgbox("Put your folder1/2 name to start training the model")
    else:

        # get_data = [,f_one, , f_two]
        cls_img, cls_label = load_img_from_folder(f_one)
        cls_img_two, cls_label_two = load_img_from_folder(f_two)

        images = np.array(cls_img + cls_img_two)
        labels = np.array(cls_label + cls_label_two)
        # print(labels)
        label_encoder = preprocessing.LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        print(labels)
        images = images / 255.0
        # Flatten the images
        images = images.reshape(images.shape[0], -1)

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.1, random_state=1
        )

        clf = SVC(kernel="linear", C=10)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        def preprocess_image(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(
                    img, (64, 64)
                )  # Resize to the same size as training images
                img = img / 255.0  #
                img = img.flatten()
                return img
            else:
                raise ValueError(f"Image not found at {image_path}")

        def predict_image(image_path, model, label_encoder):
            img = preprocess_image(image_path)
            img = np.array([img])
            prediction = model.predict(img)
            label = label_encoder.inverse_transform(prediction)
            return label[0]

        image_path = e_unseen.get()
        if image_path == "":
            easygui.msgbox(
                "ValueError: Image not found \n please check if the img is in your dir"
            )
        else:
            label = predict_image(image_path, clf, label_encoder)
            # ctypes.windll.user32.MessageBox(0 ,  , "INFO")
            easygui.msgbox("Check Log file to know Predictions", "INFO")
            with open("Results.log", "a") as file:
                file.write(
                    f"\nINFO: Model prediction is a {label}   | {datetime.datetime.now()}"
                )


win = Tk()
win.geometry("662x471")
win.title("μe - Binary Classification - V1 ")
WIN_BG = "#151515"
win.configure(bg=WIN_BG)
win.resizable(0, 0)

# LOGO
LOGO = "μe"
logo_label = Label(
    win, text=LOGO, font=("Minion Pro Med", 40, "bold"), bg=WIN_BG, fg="#CECECE"
)
logo_label.place(relx=0.01, rely=0.01)


# HEADER
HEADER = "Binary Classification."
header_label = Label(
    win, text=HEADER, font=("Consolas", 21, "bold"), bg=WIN_BG, fg="#6BA91D"
)
header_label.place(relx=0.28, rely=0.2)


# ENTRIES

# label_one = Label(
#     win, text="Label One :", font=("Consolas", 10, "bold"), bg=WIN_BG, fg="white"
# )
# label_one.place(relx=0.28, rely=0.2)
# e_label_one = Entry(
#     win,
#     width=30,
#     borderwidth=0.1,
#     bg="#3F3F3F",
#     fg="white",
#     font=("Consolas", 15, "bold"),
# )
# e_label_one.place(relx=0.28, rely=0.25)
###########
# ONE
##########
folder_one = Label(
    win, text="Folder One :", font=("Consolas", 10, "bold"), bg=WIN_BG, fg="white"
)
folder_one.place(relx=0.28, rely=0.3)
e_folder_one = Entry(
    win,
    width=30,
    borderwidth=0.1,
    bg="#3F3F3F",
    fg="white",
    font=("Consolas", 15, "bold"),
)
e_folder_one.place(relx=0.28, rely=0.35)


# label_two = Label(
#     win, text="Label Two :", font=("Consolas", 10, "bold"), bg=WIN_BG, fg="white"
# )
# label_two.place(relx=0.28, rely=0.4)
# e_label_two = Entry(
#     win,
#     width=30,
#     borderwidth=0.1,
#     bg="#3F3F3F",
#     fg="white",
#     font=("Consolas", 15, "bold"),
# )
# e_label_two.place(relx=0.28, rely=0.45)
###########
# ONE
##########
folder_two = Label(
    win, text="Folder Two :", font=("Consolas", 10, "bold"), bg=WIN_BG, fg="white"
)
folder_two.place(relx=0.28, rely=0.41)
e_folder_two = Entry(
    win,
    width=30,
    borderwidth=0.1,
    bg="#3F3F3F",
    fg="white",
    font=("Consolas", 15, "bold"),
)
e_folder_two.place(relx=0.28, rely=0.47)


unseen = Label(
    win,
    text="Unseen Data (New Data) :",
    font=("Consolas", 10, "bold"),
    bg=WIN_BG,
    fg="white",
)
unseen.place(relx=0.28, rely=0.53)
e_unseen = Entry(
    win,
    width=30,
    borderwidth=0.1,
    bg="#3F3F3F",
    fg="white",
    font=("Consolas", 15, "bold"),
)
e_unseen.place(relx=0.28, rely=0.59)


# Submit data
submit = Button(
    win,
    text="Train the model and predict unseen data",
    fg="white",
    bg="#6BA91D",
    width=46,
    command=GET,
    font=("Consolas", 10, "bold"),
)
submit.place(relx=0.28, rely=0.66)


# TIPS_NOTE
git = "Meshojs❤github"
git_label = Label(win, text=git, fg="#1DA9A1", bg=WIN_BG, font=("Consolas", 10, "bold"))
git_label.place(relx=0.01, rely=0.82)

tip = "Tips : upload folder1 and folder2 \nwhich will be used to train the model"
tip_label = Label(
    win,
    text=tip,
    bg=WIN_BG,
    font=("Consolas", 10, "bold"),
    justify=LEFT,
    fg="#6BA91D",
    anchor="w",
    width=200,
)
tip_label.place(relx=0.01, rely=0.86)

READ = "Read README.txt for more info before using !"
read_label = Label(
    win, text=READ, bg=WIN_BG, fg="#A91D3A", font=("Consolas", 10, "bold"), justify=LEFT
)
read_label.place(relx=0.01, rely=0.94)


if __name__ == "__main__":
    win.mainloop()
