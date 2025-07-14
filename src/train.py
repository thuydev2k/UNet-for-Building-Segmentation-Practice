import os
import random
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from datasets import BuildingsDataset
from utils import visualize, colour_code_segmentation, reverse_one_hot
from augmentations import get_training_augmentation, get_preprocessing, get_validation_augmentation
from unet import model
from config import x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, DEVICE, EPOCHS, TRAINING
from utils import load_class_info, filter_classes

class_names, class_rgb_values = load_class_info()
select_classes, select_class_rgb_values = filter_classes(class_names, class_rgb_values)

dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[2]

# visualize(
#     original_image = image,
#     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#     one_hot_encoded_mask = reverse_one_hot(mask)
# )

augmented_dataset = BuildingsDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset)-1)

# Different augmentations on a random image/mask pair (256*256 crop)
# for i in range(3):
#     image, mask = augmented_dataset[random_idx]
#     visualize(
#         original_image = image,
#         ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#         one_hot_encoded_mask = reverse_one_hot(mask)
#     )

# define loss function
loss = smp.utils.losses.DiceLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# define optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.00008),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists('../input/unet-for-building-segmentation-pytorch/best_model.pth'):
    model = torch.load('../input/unet-for-building-segmentation-pytorch/best_model.pth', map_location=DEVICE)

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# Get train and val dataset instances
train_dataset = BuildingsDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

valid_dataset = BuildingsDataset(
    x_valid_dir, y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

def train_model(model):
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

    return train_logs_list, valid_logs_list

if __name__ == '__main__':
    train_logs_list, valid_logs_list = train_model(model)

    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.T

    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=21)
    plt.ylabel('IoU Score', fontsize=21)
    plt.title('IoU Score Plot', fontsize=21)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('iou_score_plot.png')
    plt.show()

    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=21)
    plt.ylabel('Dice Loss', fontsize=21)
    plt.title('Dice Loss Plot', fontsize=21)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('dice_loss_plot.png')
    plt.show()