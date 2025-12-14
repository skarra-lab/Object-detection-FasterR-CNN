import torchvision
from torchvision import transforms
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_path = r"C:\Users\User2\OneDrive\Pictures\63345ee-7fb-5401-0fda-022c6efc33f3_Kindness_3.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image)

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

with torch.no_grad():
    outputs = model([image_tensor])

boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']

fig, ax = plt.subplots(1, figsize=(12, 28))
ax.imshow(image)

for box, score in zip(boxes, scores):
    if score > 0.8:
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1),x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.show()
                                 
