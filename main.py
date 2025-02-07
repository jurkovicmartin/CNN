import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from cnn import *
from dataset import Unlabeled_dataset


def show_tensor_images(images: list[torch.Tensor], titles: list[str], batch: int =None):
    """Show 4 tensor images with matplotlib.

    Args:
        images (list[torch.Tensor]): images
        titles (list[str]): titles of images (classes)
        batch (int, optional): Display also batch in the title. Defaults to None.
    """
    fig, axes = plt.subplots(2, 2)

    if batch:
        fig.suptitle(f"{batch}. Batch")

    for ax, img, title in zip(axes.flat, images, titles):
        img = img.numpy()
        # Undo normalization
        img = (img * 0.5) + 0.5
        # Torch (channels, height, width)
        # Pyplot (height, width, channels)
        img = img.transpose((1, 2, 0))

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def determine_class(prediction: int) -> str:
    """Converts class index in string.

    Args:
        prediction (int): index of output neuron with the biggest activation

    Returns:
        str: class title
    """
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
    # create_model(10, 0.01, 0.001, True)
    # return

    BATCH_SIZE = 4
    batches_show = 5

    ### LOADING DATA

    transform = transforms.Compose([
        # In case some image has different size
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    images = Unlabeled_dataset("data/seg_pred", transform=transform)

    dataloader = DataLoader(images, batch_size=BATCH_SIZE, shuffle=True)

    ### MODEL USAGE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = load_model(device)
    print(model)

    batches = 0
    for images in dataloader:
        if batches == batches_show:
            return
    
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

            # Showing the images with their predictions as titles        
            titles = [determine_class(torch.argmax(o)) for o in outputs]
            images = images.to("cpu")
            show_tensor_images(images, titles, batches + 1)

            batches += 1
    

if __name__ == "__main__":
    main()