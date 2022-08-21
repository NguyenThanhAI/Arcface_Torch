import argparse
import os

import math

import random

from itertools import groupby

from PIL import Image

from tqdm import tqdm

import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import timm

from utils_fn import enumerate_images


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    print(f'Setting all seeds to be {seed} to reproduce...')

seed_everything(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceDataset(Dataset):
    def __init__(self, images_dir: str, subset: str="train") -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
        self.transfrom = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(os.path.normpath(image).split(os.sep)[-2])

        #img = torchvision.io.read_image(image)
        img = Image.open(image).convert("RGB")
        
        #img.div_(255).sub_(0.5).div_(0.5)
        img = self.transfrom(img)

        return img, label


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        #one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class FaceModel(nn.Module):

    def __init__(self, num_classes=10177):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*(list(model.children())[:-1]))
        
        self.pooling = GeM()

        in_features = model.fc.in_features
        self.fc = ArcMarginProduct(in_features=in_features, out_features=num_classes)

    def forward(self, images, labels):
        x = self.backbone(images)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        output = self.fc(x, labels)

        return output

    def extract(self, images):
        x = self.backbone(images) 
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return x


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba_train_val_1\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba_train_val_1\val")
    parser.add_argument("--model_dir", type=str, default=r"D:\Face_Datasets\CelebA_Models")
    parser.add_argument("--checkpoint_pattern", type=str, default=r"checkpoint")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()

    return args


def train_epoch(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: optim.Optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(images, labels)
        loss = loss_fn(pred, labels)

        loss.backward()

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print("Loss: {}, [{}/{}]".format(loss, current, size))


def val_loop(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images, images)

            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
    test_loss /= num_batch
    correct /= size
    print("Accuracy: {}%, Avg loss: {}".format(correct * 100, test_loss))
    return correct, test_loss


def save_model(model: nn.Module, accuracy: float, loss: float, epoch: int, save_path: str):
    torch.save({"weights": model.state_dict(),
                "accuracy": accuracy,
                "loss": loss,
                "epoch": epoch}, save_path)
    print("Save model with accuracy: {}, loss {} at epoch: {}".format(accuracy, loss, epoch))


if __name__ == "__main__":

    args = get_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    model_dir = args.model_dir
    checkpoint_pattern = args.checkpoint_pattern
    pretrained = args.pretrained
    num_epochs  = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    train_dataset = FaceDataset(images_dir=train_dir)
    val_dataset = FaceDataset(images_dir=val_dir)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = FaceModel()

    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=torch.device("cpu"))["weights"])
    model.to(device=device)
    print("Model: {}".format(model))


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    max_accuracy = -np.inf
    save_path = os.path.join(model_dir, checkpoint_pattern + ".pth")
    for t in range(num_epochs):
        print("Epoch {}\n-------------------------------------------------".format(t + 1))
        train_epoch(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        val_acc, val_loss = val_loop(dataloader=val_dataloader, model=model, loss_fn=loss_fn)
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            save_model(model=model, accuracy=val_acc, loss=val_loss, epoch=t+1, save_path=save_path)
    print("Done")
    