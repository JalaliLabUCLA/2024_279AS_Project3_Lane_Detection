import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model import TwinLite as net
import cv2
import sys

def Run(model,img):
    img = cv2.resize(img, (640, 360))
    img_rs=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0=img_out[0]
    x1=img_out[1]

    _,da_predict=torch.max(x0, 1)
    _,ll_predict=torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255
    img_rs[DA>100]=[255,0,0]
    img_rs[LL>100]=[0,255,0]
    
    return img_rs


model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
#model.load_state_dict(torch.load('../Weights/TwinNet/digital_16/model_99.pth',map_location='cpu'))
model.load_state_dict(torch.load(sys.argv[1]))

model.eval()

input_dir = '../test_images/digital/images'
input_dir = sys.argv[2]

output_dir = '../test_images/digital/twinlitenet'
output_dir = sys.argv[3]

image_list=os.listdir(input_dir)
for i, imgName in enumerate(image_list):
    img = cv2.imread(os.path.join(input_dir,imgName))
    img=Run(model,img)
    cv2.imwrite(os.path.join(output_dir,imgName),img)