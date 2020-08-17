import os
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from util.tools import *

class dataset(Dataset):

    def __init__(self, image_root, label_root, img_x, img_y):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """
        self.images_path = image_root
        self.labels_path = label_root
        self.data_len = 0
        self.images = []
        self.labels = open(self.labels_path, "r").readlines()
        self.transform = transforms.Compose([
            transforms.Resize((img_x, img_y)),  
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        for file in self.labels:
            self.data_len += 1
            tem = file.split(" ")[0]
            temp = tem.split("-")
            self.images.append(self.images_path + temp[0] + '/' + temp[0] + "-" + temp[1] + "/" + tem + ".png")

    def __len__(self):
        """return number of points in our dataset"""
        return(self.data_len)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        img = self.images[idx]
        label = self.labels[idx].split(" ")[-1]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)
        return(img, label[:-1])


def loader_param(batch_size=8):
    img_x = 32
    img_y = 32
    batch_size = 8
    return(img_x, img_y, batch_size)

def word_rep(word, letter2index, max_out_chars, device = 'cpu'):
    pad_char = '-PAD-'
    rep = torch.zeros(max_out_chars).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        rep[letter_index] = pos
    pad_pos = letter2index[pad_char]
    rep[letter_index+1] = pad_pos
    return(rep, len(word))

def words_rep(labels_str, max_out_chars = 20, batch_size = 8, eng_alpha2index = None):
    words_rep = []
    output_cat = None
    lengths_tensor = None
    lengths = []
    for i, label in enumerate(labels_str):
        rep, lnt = word_rep(label, eng_alpha2index, max_out_chars)
        words_rep.append(rep)
        if lengths_tensor is None:
            lengths_tensor = torch.empty(len(labels_str), dtype= torch.int)
        if output_cat is None:
            output_cat_size = list(rep.size())
            output_cat_size.insert(0, len(labels_str))
            output_cat = torch.empty(*output_cat_size, dtype=rep.dtype, device=rep.device)
#             print(output_cat.shape)

        output_cat[i, :] = rep
        lengths_tensor[i] = lnt
        lengths.append(lnt)
    return(output_cat, lengths_tensor)


class IAMDataSet(Dataset):
    def __init__(self, image_root, label_root, target_size=(200, 32), characters="'>' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\"!#&\'()*+,-./0123456789:;?'", transform=None):
        super(IAMDataSet, self).__init__()
        self.images_path = image_root
        self.labels_path = label_root
        self.target_size = target_size
        self.height = self.target_size[1]
        self.width = self.target_size[0]
        self.characters = characters
        self.imgs = []
        self.lexicons = []
        self.parse_txt()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path, lexicon_index = self.imgs[item].split(" ")
        #print(img_path)
        lexicon = self.lexicons[int(lexicon_index)].split(" ")[-1]
        lexicon = lexicon[:-1]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = img.shape
        if (img_size[1] / (img_size[0] * 1.0)) < 6.4:
            img_reshape = cv2.resize(img, (int(32.0 / img_size[0] * img_size[1]), self.height))
            mat_ori = np.zeros((self.height, self.width - int(32.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
            out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
        else:
            out_img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            out_img = np.asarray(out_img).transpose([1, 0, 2])

        label = [self.characters.find(c) for c in lexicon]
        if self.transform:
            out_img = self.transform(out_img)
        return out_img, label

    def parse_txt(self):
        # self.imgs = open(os.path.join(self.dataset_root, self.anno_txt_path), 'r').readlines()
        # self.lexicons = open(os.path.join(self.dataset_root, self.lexicon_path), 'r').readlines()
        self.lexicons = open(self.labels_path, "r").readlines()
        for file in range(len(self.lexicons)):
            tem = self.lexicons[file].split(" ")[0]
            temp = tem.split("-")
            self.imgs.append(self.images_path + temp[0] + '/' + temp[0] + "-" + temp[1] + "/" + tem + ".png" + " " + str(file))






if __name__ == '__main__':

    img_x, img_y, batch_size = loader_param()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])



    # train_set = dataset(image_root="/home/skyatmoon/COMP4550/IAM/words/", label_root = "/home/skyatmoon/COMP4550/IAM/words.txt", img_x = img_x, img_y = img_y)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    train_set = IAMDataSet(image_root="/home/skyatmoon/COMP4550/IAM/words/", label_root = "/home/skyatmoon/COMP4550/IAM/words.txt",
                          target_size=(300, 32), characters="'>' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\"!#&\'()*+,-./0123456789:;?'", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    train_iter = iter(train_loader)
    samples, labels, target_lengths, input_lengths = next(train_iter)

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    trial_num = np.random.randint(0, batch_size)
    imshow(samples[0])
    print(labels)



    # eng_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"!#&\'()*+,-./0123456789:;?'
    # pad_char = '-PAD-'

    # eng_alpha2index = {pad_char: 0}
    # for index, alpha in enumerate(eng_alphabets):
    #     eng_alpha2index[alpha] = index+1

    # print(eng_alpha2index)
    # input_lengths = torch.full(size=(batch_size,), fill_value=img_y, dtype=torch.int)
    # print(input_lengths)
    # for j, (images, labels_str) in enumerate(iter(train_loader)):
    #         labels, target_lengths = words_rep(labels_str, max_out_chars = 20, batch_size = batch_size , eng_alpha2index = eng_alpha2index)
    #         print(target_lengths)