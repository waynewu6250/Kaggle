import os
import numpy as py
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T

# Define our customed dataset

class DogCat(Dataset):
    def __init__(self, root, transforms=None, train=False, test=False):
        """
        Get all images and split data
        """
        self.train = train
        self.test = test

        #1. get all images path and sort
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        #train img path: data/train/cat.1.jpg
        #test img path: data/test/1.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
            
        
        #2. split the data
        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[:int(0.7*len(imgs))]
        else:
            self.imgs = imgs[int(0.7*len(imgs)):]

        #3. Transform
        if transforms is None:
            if self.train:
                self.transforms = T.Compose([T.Resize(224),
                                             T.CenterCrop(224),
                                             T.ToTensor(),
                                             T.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225])
                                            ])
            else:
                self.transforms = T.Compose([T.Resize(224),
                                             T.RandomSizedCrop(224),
                                             T.RandomHorizontalFlip(),
                                             T.ToTensor(),
                                             T.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225])
                                            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        
        return data, label

    def __len__(self):
        return len(self.imgs)
        
