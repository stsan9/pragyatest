import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_model():
    # load in pretrained resnet18 architecture
    resnet18 = models.resnet18(pretrained=True)
    # replace input and output layers to fit our dataset and problem
    conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change input layer to take a 1 dimensional image
    fc = nn.Linear(512, 7)    # change output layer to output a value per class (7)
    resnet18.conv1 = conv1
    resnet18.fc = fc
    model = resnet18
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))    # our saved parameters
    return model

# modified https://medium.com/yottabytes/a-quick-guide-on-preprocessing-facial-images-for-neural-networks-using-opencv-in-python-47ee3438abd4
def load_input_face(file, sharpen=False):
    """
    Load in our image file,
    convert to grayscale,
    identify one face in the image
    crop the image to only fit the face
    resize it to a 48x48 image and then normalize

    :param img_filename: str path to image file
    :param sharpen: sharpen the image if we want to
    """
    # load in img
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # load in face classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # identify the face
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    # crop image to show just the face
    (x, y, w, h) = faces[0]
    img = img[y:y+h, x:x+w]
    # resize to a 48x48 image
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_LINEAR)
    if sharpen: # sharpen the image
        smoothed_img = cv2.GaussianBlur(img, (0,0), 1)
        sharpened_img = cv2.addWeighted(img, 1.5, smoothed_img, -0.5, 0, smoothed_img)
        img = sharpened_img
    return img / 255

def predict_expression(model, img):
    """
    Get the model expression prediction

    :param model: trained expression classifier model
    :param img: 48x48 numpy array normalized
    """
    # move model to cpu, and make image valid torch tensor shape
    model.to('cpu')
    img = torch.tensor(img.reshape(1, 1, 48, 48)).float()

    # get predicted class label (int)
    pred = model(img)
    pred = torch.argmax(pred)
    pred = pred.item()

    # turn label to string
    expressions_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    pred_expression = expressions_dict[pred]
    return pred_expression

def show_face_and_pred(img, pred_expression):
    fig = plt.figure()
    plt.title(f'Prediction: {pred_expression}', fontsize=18)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    return fig

model = load_model()
st.title('Facial Expression Classifier')
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file != None:
    img = load_input_face(file)
    pred_expression = predict_expression(model, img)
    st.pyplot(show_face_and_pred(img, pred_expression))
