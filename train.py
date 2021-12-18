import torch
import segmentation_models_pytorch as smp
import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split
import utils
import wandb
from mydata import MyData
from torch.utils.data import DataLoader
import losses
from tqdm import tqdm
from scores import DiceScores
import numpy as np

#读取数据
def load_data(batch_size):
    train_trans= A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.RandomBrightnessContrast()
        ], p=0.3)])
    val_trans = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90()])

    df = pd.read_csv('./data/train.csv')
    names = list(df['names'].apply(lambda x:'./data/train/' + x))
    masks = list(df['masks'])
    train_set_path, val_set_path, train_mask, val_mask = train_test_split(names, masks, train_size=0.8, shuffle=True)

    train_set = MyData(train_set_path, train_mask, is_train=True, transform=train_trans)
    val_set = MyData(val_set_path, val_mask, is_train=True, transform=val_trans)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader

#读取模型
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    return model

def train_(model, train_loader, val_loader, device, epochs):
    best_score = 0

    experiment = wandb.init(project='U-net', resume='allow', anonymous='must')
    loss_fc = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.36)]
    optimizer =torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    train_epoch =smp.utils.train.TrainEpoch(
        model,
        loss=loss_fc,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True
    )
    val_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss_fc,
        metrics=metrics,
        device=device,
        verbose=True
    )

    for epoch in range(epochs):
        print(f"\nEpoch:{epoch+1}")
        train_logs =train_epoch.run(train_loader)
        val_logs = val_epoch.run(val_loader)
        experiment.log({
            'epoch': epoch + 1,
            'train dice loss': train_logs['dice_loss'],
            'train iou scores': train_logs['iou_score'],
            'val dice loss': val_logs['dice_loss'],
            'val iou score': val_logs['iou_score']
        })
        if best_score < val_logs['iou_score']:
            best_score = val_logs['iou_score']
            torch.save(model, "./models/best_score.pth")
            print("Model save!")
        if epoch+1 == epochs:
            torch.save(model, "./models/last_model.pth")
            print("Last model save!")

#验证，返回得分和损失
@torch.no_grad()
def validation(model, val_loader, device):
    model.eval()
    dice_score = 0
    dice_loss = 0
    val_len = len(val_loader)
    calc_score = DiceScores()
    calc_loss = losses.EvalLoss()

    for img,target in tqdm(val_loader):
        img, target = img.to(device), target.float().to(device)
        y_pred = model(img)
        dice_score += calc_score(y_pred, target)
        dice_loss += calc_loss(y_pred, target)

    return dice_score / val_len, dice_loss / val_len

# 训练
def train(model, train_loader, val_loader, device, epochs):
    best_score = 0
    experiment = wandb.init(project="U-net", resume="allow", anonymous="must")
    loss_fc = losses.EvalLoss()
    score_fc = DiceScores()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_scores = []
        print(f"Epoch:{epoch+1}")
        for img, target in tqdm(train_loader):#[2, 3, 512, 512],[2, 1, 512, 512]
            img, target = img.to(device), target.float().to(device)
            optimizer.zero_grad()
            output = model(img)
            train_scores.append(score_fc(output, target))
            loss = loss_fc(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_dice_score, val_eval_loss = validation(model, val_loader, device)
        scheduler.step(val_eval_loss)
        train_eval_loss = np.sum(train_losses) / len(train_losses)
        train_dice_score = np.sum(train_scores) / len(train_scores)
        print(f"Train | val_loss={val_eval_loss:.4f}, dice_score={val_dice_score:.4f}")
        if val_dice_score > best_score:
            best_score = val_dice_score
            torch.save(model.state_dict(), "./models/best_1.pth")
            print("Model save!")
        if epoch+1 == epochs:
            torch.save(model.state_dict(), "./models/last.pth")
            print("Last model save!")
        experiment.log({
            "epoch":epoch+1,
            "train dice loss": train_eval_loss,
            "train dice score": train_dice_score,
            "val dice loss":val_eval_loss,
            "val dice score":val_dice_score
        })


if __name__ == '__main__':
    utils.set_seeds()
    device = utils.try_gpu()
    train_iter,val_iter = load_data(batch_size=8)
    model = load_model()
    train(model, train_iter, val_iter, device, 10)
