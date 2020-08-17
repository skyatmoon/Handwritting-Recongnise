import torch.backends.cudnn as cudnn
import math
import time
import datetime
import argparse
import os
from torch.nn import CTCLoss
from vision.network import CRNN
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from util.data_loader import RegDataSet
from util.tools import *
from IAMloade import *
import matplotlib.pyplot as plt
import pandas as pd


epoch=np.loadtxt('enum')
trainloss=np.loadtxt('trainloss')
valloss=np.loadtxt('valloss')

plt.plot(epoch, valloss, color = "r", marker = ".")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("val-loss")
plt.savefig("val-loss.png")


# plt.plot(epoch, trainloss, color="b", marker = ".")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("train-loss")
# plt.savefig("train-loss.png")