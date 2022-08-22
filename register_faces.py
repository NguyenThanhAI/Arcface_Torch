import os
import argparse
from typing import List, Dict
from itertools import groupby

from PIL import Image

import numpy as np
import cv2
import torch
from yaml import parse

from backbones import get_model
import facenet_pytorch


from mtcnn import MTCNN
from utils_fn import enumerate_images
from face_database import FaceRecognitionDataBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(100)

def get_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\hand_faces")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--num_images_per_id", type=int, default=5)
    parser.add_argument("--db_file", type=str, default="face_db.db")
    parser.add_argument("--network", type=str, default="r50")
    parser.add_argument("--weights", type=str, default=r"C:\Users\Thanh\Downloads\backbone.pth")
    parser.add_argument("--model_type", type=str, default="arcface")

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()

    images_dir = args.images_dir
    num_images_per_id = args.num_images_per_id
    db_file = args.db_file
    network = args.network
    weights = args.weights

    model = get_model(name=network)
    model.load_state_dict(torch.load(weights))
    model.eval()
    model.to(device)

    mtcnn = MTCNN()
    db = FaceRecognitionDataBase(db_file)

    images_list: List[str] = enumerate_images(images_dir=images_dir)

    #images_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
    images_list.sort()

    id_to_images: Dict[str, List[str]] = {}
    for keys, items in groupby(images_list, key=lambda x: x.split(os.sep)[-2]):
        id_to_images[keys] = list(items)
    

    for identity in id_to_images:

        #images = np.random.choice(id_to_images[identity], size=num_images_per_id, replace=False).tolist()
        images = id_to_images[identity][:num_images_per_id]
        features = []

        for image in images:
            img = Image.open(image).convert("RGB")
            face = mtcnn.align(img=img)
            if face is None:
                continue
            img = np.array(face)
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            #print(img.shape)
            img.div_(255).sub_(0.5).div_(0.5)

            img = img.to(device=device)
            #print(img.shape)

            with torch.no_grad():
                feature = model(img).cpu().numpy()[0]
            
            feature = feature / np.linalg.norm(feature)

            features.append(feature)

        if len(features) > 0:
            print("Number of features: {}".format(len(features)))
            features = np.stack(features, axis=0)
            print("Number of features: {}".format(features.shape[0]))
            db.add_persons(info=identity, features=features)
            last_id = db.get_latest_id()[0]
            print("id: {}, info: {}".format(last_id, identity))

        else:
            print("No face to add")
