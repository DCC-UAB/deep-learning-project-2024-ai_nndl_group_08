import torchvision

from ConTextDataset import ConTextDataset
from ConTextTransformer import ConTextTransformer
from train import evaluate as evaluate

import torch

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

def test_saved_model(model_path, test_loader):
    # Initialize the model
    model = ConTextTransformer(image_size=input_size, num_classes=28, channels=3, dim=256, depth=3, heads=4,mlp_dim=512)

    # Load the model state from the saved file
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize a list to keep track of the test loss
    test_loss_history = []

    # Evaluate the model and print the accuracy
    accuracy = evaluate(model, test_loader, test_loss_history)
    print(f'Test accuracy of the saved model: {accuracy:.2f}%')

    return accuracy


data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_set  = ConTextDataset(json_file, img_dir, txt_dir, "test", data_transforms_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

model_path = 'trained_transformer_model.pth'
acc = test_saved_model(model_path, test_loader)

print("Accuracy: ", acc)




