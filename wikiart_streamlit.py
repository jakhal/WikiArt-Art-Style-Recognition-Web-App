# author: Jakob Halswick

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import os

st.title("Kunststilerkennung mit neuronalen Netzen")
st.write("Modell basiert auf ResNet50 (vortrainiert mit ImageNet, Finetuning mit WikiArt Datensatz)")
st.write("Zur Vorhersage des Kunststils Bild hochladen oder mit Kamera aufnehmen.")

img_size = 512
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# upload image
file_up = st.file_uploader("Bild hochladen", type = "jpg")


class_names = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get current working directory to find weights of model
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def predict(image):
    model = models.resnet50(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(os.path.join(__location__, 'resnet50_512_transplus.pth'), map_location=device))
    model = model.to(device)
    
    transform = data_transforms['test']

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    out = model(batch_t)

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(class_names[idx], prob[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Bild hochgeladen.', use_column_width = True)
    st.write("")
    st.write("Lädt...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with probabilities
    for i in labels:
        st.write(i[0], ",   Wahrscheinlichkeit: ", i[1])

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture, caption = 'Bild aufgenommen.', use_column_width = True)
    image = Image.open(picture)
    st.write("")
    st.write("Lädt...")
    labels = predict(picture)

    # print out the top 5 prediction labels with probabilities
    for i in labels:
        st.write(i[0], ",   Wahrscheinlichkeit: ", i[1])


