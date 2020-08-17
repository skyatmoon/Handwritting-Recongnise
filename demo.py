import os
import argparse
import torch
import cv2
from torchvision.transforms import transforms
import time
from vision.network import CRNN
from util.tools import process_img, decode_out
from PIL import Image
import matplotlib.pyplot as plt

loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()



parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/Final-crnn.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_file', default='result.txt', type=str, help='file to save results')
parser.add_argument('--width', default=200, type=int, help="input image width")
parser.add_argument('--height', default=32, type=int, help="input image height")
parser.add_argument('--cpu', default=False, help='Use cpu inference')
parser.add_argument('--characters', default='-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"!#&\'()*+,-./0123456789:;?', type=str, help="characters")
parser.add_argument('--input_path', default='./test/', type=str, help="image or images dir")
args = parser.parse_args()

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

def img_cro(img,pieces,points,name):

    for i in range(len(points)-1):
        cropped = img[0:int(img.shape[0]), (int((img.shape[1]/pieces)*points[i])): (int((img.shape[1]/pieces)*points[i+1]))] # 裁剪坐标为[y0:y1, x0:x1]
        cv2.imwrite("./res/"+ str(name) + "_" + str(i) + ".png", cropped)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    net = CRNN(len(args.characters))
    device = torch.device("cpu" if args.cpu else "cuda")
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device(device)))
    net.eval()

    input_path = args.input_path
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    name = 0
    for img_path in image_paths:
        begin = time.time()
        print("recog {}".format(img_path))
        image = cv2.imread(img_path)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img = process_img(image, args.height, args.width, transform)
        #imshow(img)

        net_out = net(img)
        _, preds = net_out.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        print(preds)

        cut_points = []
        length = 0

        lab2str, cut_points, length = decode_out(preds, args.characters)
        cut_points.append(length)

        print(length)
        print(cut_points)
        print(lab2str)
        img_cro(image,length,cut_points,name)
        end = time.time()
        name =name +1

    print("Done!!!")