import numpy as np
import time
import os

import numpy as np 
#import pandas as pd 
import pickle
import torch
from PIL import Image
import random
import math
#import matplotlib.pyplot as plt
#from torchvision.utils import make_grid

from . import GoogLeNet

print(os.getcwd())

def preprocess_image(image_path):
	'''입력받은 이미지 전처리 후 4차원의 텐서타입으로 변경'''
	#img = load_img(image_path, target_size=(img_height, img_width)) # (400, 381)
	'''The img_to_array() function adds channels: x.shape = (224, 224, 3) for RGB and (224, 224, 1) for gray image'''
	img = Image.open(image_path).convert("L")
	original_width, original_height = img.size
	
	resize_img = img.resize((224, 224))
	re_img_arr=np.asarray(resize_img)
	arr = (re_img_arr - re_img_arr.min()) / (re_img_arr.max() - re_img_arr.min())
	
	torch_img = torch.from_numpy(arr).clone()
	torch_img_add_demension = torch.unsqueeze(torch_img,0)
	torch_img_add_demension2 = torch.unsqueeze(torch_img_add_demension,0)

	return torch_img_add_demension2, original_width, original_height

def deprocess_image():

	return 0 


def main(target_img_path):
	# image 파일명
	target_img_path = target_img_path.split('/')[-1]
	# image 경로
	target_image_path = 'pyflask/static/pyimages/'+ target_img_path 			# 타깃 이미지 

	# 이미지 전처리
	target_torch_image = preprocess_image(target_image_path)[0] # creates img to a constant tensor

	# 학습 모델 불러오기
	load_model = GoogLeNet.GoogLeNet(4)
	path = "pyflask/model/"
	load_model.load_state_dict(torch.load(path+"model_state_dict_epoch10_2.pt", map_location='cpu'),strict=False)
	load_model.eval()

	# 예측하기
	pred_list = []

	device = torch.device('cpu')
	dtype = torch.float
	
	x = target_torch_image.to(device=device, dtype=dtype)  

	scores = load_model(x)
	_, preds = scores.max(dim=1)
	pred_list += preds.to("cpu")

	pred_class = pred_list[0].numpy()
	
	print(pred_class)
	
	return pred_class

if __name__ == "__main__":
	main()