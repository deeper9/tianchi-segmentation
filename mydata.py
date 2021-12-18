from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import utils

class MyData(Dataset):
    def __init__(self, paths, rles, transform, is_train=True):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.is_train = is_train

        self.as_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        img = cv2.imread(self.paths[item])
        if self.is_train:
            mask = utils.rle_decode(self.rles[item])
            augments =self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None]
        else:
            return self.as_tensor(img),""

    def __len__(self):
        return len(self.paths)
