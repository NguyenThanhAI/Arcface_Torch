import os
import argparse
from typing import List
from tqdm import tqdm

import numpy as np

import cv2

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn


import torchattacks

from torchattacks.attack import Attack
from torchattacks import VANILA, GN, FGSM, BIM, CW, RFGSM, PGD, PGDL2, EOTPGD, TPGD, FFGSM, \
    MIFGSM, APGD, APGDT, FAB, Square, AutoAttack, OnePixel, DeepFool, SparseFool, DIFGSM, UPGD, TIFGSM, Jitter, \
    Pixle

from backbones import get_model
from utils_fn import enumerate_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default=r"D:\Face_Datasets\CelebA_Models\checkpoint.pth")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    #parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\facenet_pytorch_torchattacks_images")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\choose_train_torchattacks_images")

    args = parser.parse_args()

    return args


'''class FaceDataset(Dataset):
    def __init__(self, images_dir: str) -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(image.split(os.sep)[-2])

        img = cv2.imread(image)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        img.div_(255)

        return img, label'''

class FaceDataset(Dataset):
    def __init__(self, images_dir: str) -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
        identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], self.images_list))))
        identities.sort()
        identities_to_id = dict(zip(identities, list(range(len(identities)))))
        self.images_to_groundtruth_id = dict(zip(self.images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], self.images_list))))
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(self.images_to_groundtruth_id[image])
        #print(image, label)

        img = cv2.imread(image)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        #img.div_(255).sub_(0.5).div_(0.5)
        img.div_(255)

        return img, label


class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        for layer in self.backbone.parameters():
            layer.requires_grad = False
        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output


if __name__ == "__main__":
    args = get_args()

    weights = args.weights
    images_dir = args.images_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    '''images_list = enumerate_images(images_dir=images_dir)
    images_list.sort(key=lambda x: int(os.path.normpath(x).split(os.sep)[-2]))
    image_ord_to_image_name = dict(zip(range(len(images_list)), images_list))'''

    images_list = enumerate_images(images_dir=images_dir)
    identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    identities.sort()
    identities_to_id = dict(zip(identities, list(range(len(identities)))))
    id_to_identities = {v: k for k, v in identities_to_id.items()}
    images_to_groundtruth_id = dict(zip(images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], images_list))))
    image_ord_to_image_name = dict(zip(range(len(images_list)), images_list))

    #model = torchvision.models.resnet18(num_classes=10177)
    model = FaceModel(model_name="r18", num_classes=len(identities))
    model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])

    model.eval()
    model.to(device=device)

    attack_types: List[Attack] = [
    RFGSM(model, eps=8/255, alpha=2/255, steps=100)
    ]

    dataset = FaceDataset(images_dir=images_dir)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    for images, labels in dataloader:
        print(images.min(), images.max())

    for atk in tqdm(attack_types):

        atk.set_return_type("int")

        atk_name = atk.__class__.__name__
        print("Attack name: {}".format(atk_name))
        atk_save_dir = os.path.join(save_dir, atk_name)
        if not os.path.exists(atk_save_dir):
            os.makedirs(atk_save_dir, exist_ok=True)

        try:
            atk.save(data_loader=dataloader, save_path=os.path.join(atk_save_dir, "adv_data.pt"), verbose=True)
        except Exception as e:
            print("Error: {}".format(e))
            continue

        adv_images, adv_labels = torch.load(os.path.join(atk_save_dir, "adv_data.pt"))

        print(adv_images.min(), adv_images.max())

        adv_dataset = TensorDataset(adv_images)

        for i, adv_image in enumerate(adv_dataset):
            img = adv_image[0].numpy()
            img = np.transpose(img, (1, 2, 0))
            #print(img.shape)
            image_path = image_ord_to_image_name[i]
            label_id = images_to_groundtruth_id[image_path]
            label = id_to_identities[label_id]
            #print(label)
            img_save_dir = os.path.join(atk_save_dir, label)
            img_save_name = os.path.basename(image_path)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir, exist_ok=True)
            img_save_path = os.path.join(img_save_dir, img_save_name)
            cv2.imwrite(img_save_path, img[:, :, ::-1])