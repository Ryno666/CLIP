import torch
from torch import nn
import clip
from PIL import Image

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

image = preprocess(Image.open("ACE.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a man", "a dog", "a cat", "one piece", "Japan", "Ace"]).to(device)

with torch.no_grad():
    logits_image, logits_text = model(image, text)
    probs = logits_image.softmax(dim=-1).cpu().numpy()
    
    
print(f"imagesize : {image.shape}, text size : {text.shape}")
print(f"image : {logits_image}")
print("label probs : ", probs)
# print(type(probs))
# print(f"predicted label is {text.item()[int(probs.item())]}")



# import torch
# import clip
# from PIL import Image
# import numpy as np
# from torch import nn
 
# device = "cuda:3" if torch.cuda.is_available() else "cpu"


# # 加载预训练好的模型
# model, preprocess = clip.load("ViT-B/32", device=device)
# # if torch.cuda.device_count() > 1:
# #     model = nn.DataParallel(model)
# # 读取艾斯的图片和候选类别文字
# image = preprocess(Image.open("bb.jpeg")).unsqueeze(0).to(device)
# text = clip.tokenize(["cute", "handsome"]).to(device)
# cls_output = ["cute", "handsome"]

# with torch.no_grad():
#     # 计算每一张图像和每一个文本的相似度值
#     logits_per_image, logits_per_text = model(image, text)
 
#     # 对该image与每一个text的相似度值进行softmax
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
# print("Label probs:", probs)
# # print(type(probs))
# max_idx = np.argmax(probs)
# # print(max_idx)
# print(cls_output[max_idx])