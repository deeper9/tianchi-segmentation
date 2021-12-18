import pandas as pd
import shutil

img_paths = './data/train'
mask_path = "./data/train.csv"
df = pd.read_csv(mask_path)
index = df["masks"].notnull()
names = df["names"][index]
masks = df["masks"][index]
names = names[:100]
masks = masks[:100]

for n in names:
    src = img_paths + "/" + n
    dst = "./data/train_a/" + n
    shutil.copyfile(src, dst)

df1 = pd.DataFrame({"names":names, "masks":masks}, index=None)
df1.to_csv("./data/train_a.csv",index=0)