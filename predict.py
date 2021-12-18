import torch
import cv2
from torchvision import transforms
import train
import os
import scores
import utils


as_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
device = utils.try_gpu()
state = torch.load("./models/best.pth")
model = train.load_model()
model.load_state_dict(state)
model.to(device)

calc_score = scores.DiceScores()
@torch.no_grad()
def predict(img_path):
    img = cv2.imread(img_path)
    img = as_tensor(img)
    img = torch.unsqueeze(img, dim=0)
    mask = model.predict(img.to(device))
    pr_mask = (mask.squeeze().cpu().detach().numpy().round())

    return pr_mask

def show():
    root = "./data/test/"
    img_names = os.listdir(root)
    for name in img_names:
        img_path = root + name
        mask = predict(img_path)
        utils.show_image(img_path, mask, is_path=True)
show()
