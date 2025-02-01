import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from cnn import *
from dataset import Unlabeled_dataset


def show_tensor_image(img: torch.Tensor, title: str):
    img = img.numpy()
    img = (img * 0.5) + 0.5
    # Torch (channels, height, width)
    # Pyplot (height, width, channels)
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def determine_class(prediction: int):
    if prediction == 0:
        return "buildings"
    elif prediction == 1:
        return "forest"
    elif prediction == 2:
        return "glacier"
    elif prediction == 3:
        return "mountain"
    elif prediction == 4:
        return "sea"
    else:
        return "street"
    


def main():
    # create_model(20, 0.001, True)
    # return

    batch_size = 10
    batches_show = 1

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    images = Unlabeled_dataset("data/seg_pred", transform=transform)

    dataloader = DataLoader(images, batch_size=batch_size, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = load_model(device)

    batches = 0
    for images in dataloader:
        if batches == batches_show:
            return
        
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            images = images.to("cpu")
            
            for i in range(len(images)):
                prediction = torch.argmax(outputs[i])
                title = determine_class(prediction)
                show_tensor_image(images[i], title)

            batches += 1

    

if __name__ == "__main__":
    main()