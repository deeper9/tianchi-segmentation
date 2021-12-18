from tqdm import tqdm
import cv2
from torchvision import transforms
import torch
import utils
import train
import pandas as pd

device = utils.try_gpu()

as_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
state = torch.load("./models/best5.pth")
model = train.load_model()
model.load_state_dict(state)
model.to(device)

#计算测试集得分
subm = []
df = pd.read_csv("./data/test_a_samplesubmit.csv",sep='\t',names=['name','mask'])
names = df["name"]
for idx, name in enumerate(tqdm(names)):
    img_path = "./data/test_a/" + name
    image = cv2.imread(img_path)
    image = as_tensor(image)
    with torch.no_grad():
        image = image.to(device)[None]
        score1 = model.predict(image).cpu().numpy()

        score2 = model.predict(torch.flip(image, [0, 3]))
        score2 = torch.flip(score2, [3, 0]).cpu().numpy()

        score3 = model.predict(torch.flip(image, [0, 2]))
        score3 = torch.flip(score3, [2, 0]).cpu().numpy()

        score = (score1 + score2 + score3) / 3.0
        score_sigmoid = (score[0][0] > 0.36).astype("int32")
        subm.append([img_path.split('/')[-1], utils.rle_encode(score_sigmoid)])
        if (idx + 1) % 100 == 0:
            utils.show_image(img_path, score_sigmoid, is_path=True)

subm = pd.DataFrame(subm)
subm.to_csv('./data/test_a_tta_submit.csv', index=None, header=None, sep='\t')