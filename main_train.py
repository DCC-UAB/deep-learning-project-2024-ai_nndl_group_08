import torchvision

import time

from ConTextDataset import ConTextDataset
from ConTextTransformer import ConTextTransformer

import torch

from train import evaluate as evaluate
from train import train_epoch as train_epoch


# Use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)

# File containing image names and their target classes
json_file = './annotations/split_0.json'

# Directory of images
img_dir = "./images/data/JPEGImages/"

# Directory of OCR annotations for each image
txt_dir = "./ocr_labels/"

input_size = 256

# Defining image transformations for preprocessing
data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Generating the train and validation loaders
train_set = ConTextDataset(json_file, img_dir, txt_dir, "train", data_transforms_train)
validation_set = ConTextDataset(json_file, img_dir, txt_dir, "validation", data_transforms_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=8)



N_EPOCHS = 30
start_time = time.time()

# Create new, untrained model
model = ConTextTransformer(image_size=input_size, num_classes=28, channels=3, dim=256, depth=3, heads=4, mlp_dim=512)
model.to(device)
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)

train_loss_history, test_loss_history = [], []
best_acc = 0.
losses = []
accs = []

'''
    Train the model on the train_set
    Save the model that provides the best accuracy in the specified file
    Using the validation set for evaluation
'''
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    loss = train_epoch(model, optimizer, train_loader, train_loss_history)
    losses.extend(loss)
    acc = evaluate(model, validation_loader, test_loss_history)
    accs.append('{:4.2f}'.format(100.0 * acc))
    if acc>best_acc:
        torch.save(model.state_dict(), 'trained_transformer_model.pth')
        best_acc = acc
    scheduler.step()

print("Accuracy: ", accs)
print("Losses: ", losses)


''' 
    Import the best model from earlier and train in again on the validation set
    In this way, the model will learn from as many samples as possible 
'''
model_path = 'trained_transformer_model.pth'
model = ConTextTransformer(image_size=input_size, num_classes=28, channels=3, dim=256, depth=3, heads=4, mlp_dim=512)
model.load_state_dict(torch.load(model_path))
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
optimizer = torch.optim.Adam(params_to_update, lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)

for epoch in range(1, 10):
    print('Epoch:', epoch)
    train_epoch(model, optimizer, validation_loader, train_loss_history)
    scheduler.step()

# Now save the model again
torch.save(model.state_dict(), 'trained_transformer_model.pth')