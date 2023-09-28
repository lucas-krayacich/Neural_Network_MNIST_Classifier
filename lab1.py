import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import torch as torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from model import autoencoderMLP4Layer
import argparse


# prompt and open
def get_image_with_input():
    print("enter an integer between  0 and 59,9999")
    inputted_index = int(input())
    if inputted_index > 0 and inputted_index < 599999:
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = MNIST('./data/mnist', train=True, download=False, transform=train_transform)
        plt.imshow(train_set.data[inputted_index], cmap='gray')
        plt.show()
    else:
        print("need a small or positive integer")


def eval_and_show(model, device, weight_path):
    print("FOR BASIC EVAL -- enter an integer between  0 and 59,9999")
    input_index = int(input())

    model.load_state_dict(torch.load(weight_path))
    model.eval()

    if input_index > 0 and input_index < 599999:
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = MNIST('./data/mnist', train=True, download=False, transform=train_transform)
        device = torch.device("cpu")

        # Get the chosen image by its index
        input_image, lbl = train_set[input_index]
        input_image = torch.tensor(input_image, dtype=torch.float32).to(device)
        # input_image = input_image.clone().detach()

        # Flatten the image to match the input format expected by your model
        preprocessed_image = input_image.view(1, -1)

        # Pass the preprocessed image through your model for evaluation
        with torch.no_grad():
            output_image = model(preprocessed_image.view(-1, 28 * 28))

        plt.subplot(1, 2, 1)
        plt.imshow(train_set.data[input_index], cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(output_image.view(28, 28), cmap='gray')
        plt.title("Reconstructed Image")

        plt.show()
    else:
        print("Invalid number inputted.")


def noise_it_up(image):
    noise = torch.rand(image.size()) * 0.4
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0.1, 0.9)

    return noisy_image


def eval_noise_show(model, device, weight_path):
    print("FOR NOISY EVAL -- enter an integer between  0 and 59,9999")
    input_index = int(input())
    saved_weights_path = weight_path
    model.load_state_dict(torch.load(saved_weights_path))
    model.eval()

    if input_index > 0 and input_index < 599999:
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = MNIST('./data/mnist', train=True, download=False, transform=train_transform)

        # Get the chosen image by its index
        # input_image = train_set.data[input_index]
        input_image, lbl = train_set[input_index]
        noised_image = noise_it_up(input_image)
        noised_image_input = torch.tensor(noised_image, dtype=torch.float32).to(device)

        # Flatten the image to match the input format expected by your model
        preprocessed_image = noised_image_input.view(1, -1)

        # Pass the preprocessed image through your model for evaluation
        with torch.no_grad():
            output_image = model(preprocessed_image.view(-1, 28 * 28))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(train_set.data[input_index], cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(noised_image.squeeze(), cmap='gray')
        plt.title("Noised Image")

        plt.subplot(1, 3, 3)
        plt.imshow(output_image.view(28, 28).cpu().numpy(), cmap='gray')
        plt.title("Reconstructed Image")

        plt.show()
    else:
        print("Invalid number inputted.")

def interpolate(tensor1, tensor2, n):
    interpolate_list = []
    for i in range(n+1):
        interpolation = (i/n)*tensor1 + ((1-(i/n))*tensor2)
        interpolate_list.append(interpolation)
    return interpolate_list

def bottleneck_interpolation(model, device, weight_path):
    print(" FOR BOTTLENECK IMAGE 1 -- enter an integer between  0 and 59,9999")
    img1_index = int(input())

    print("FOR BOTTLENECK IMAGE 2 --enter another integer between  0 and 59,9999")
    img2_index = int(input())

    print("enter the number of linear interpolations to occur:")
    n = int(input())

    if img1_index > 59999 or img1_index < 1 or img2_index > 59999 or img2_index < 1:
        print("invalid inputs")
        return

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=False, transform=train_transform)

    img1, lbl1 = train_set[img1_index]
    img2, lbl2 = train_set[img2_index]

    # # declare / instantiate model
    model = model
    # saved_weights_path = "./data/MLP.8.pth"
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # pre-process both images
    processed_img1 = torch.tensor(img1, dtype=torch.float32).to(device)
    processed_img1 = processed_img1.view(1, -1)
    with torch.no_grad():
        bottleneck_tensor1 = model.encode(processed_img1.view(-1, 28 * 28))

    processed_img2 = torch.tensor(img2, dtype=torch.float32).to(device)
    processed_img2 = processed_img2.view(1, -1)
    with torch.no_grad():
        bottleneck_tensor2 = model.encode(processed_img2.view(-1, 28 * 28))

    # interpolate images

    # Feed all interpolated images through encoder function within model
    current_bottleneck = bottleneck_tensor2

    # plt.figure((2 + n, 2 + n))
    plt.subplot(1, 3 + n, 2 + n)
    plt.imshow(img1.view(28, 28), cmap='gray')
    plt.title("Image Goal")

    interpolate_list = interpolate(bottleneck_tensor1, bottleneck_tensor2, n)

    for i, j in enumerate(interpolate_list):
        reconstructed_image = model.decode(j).view(28, 28).detach().numpy()
        plt.subplot(1, 3 + n, i + 2)
        plt.imshow(reconstructed_image, cmap="gray")

    plt.subplot(1, 3 + n, 1)
    plt.imshow(img2.view(28, 28), cmap='gray')
    plt.title("Image 1")

    plt.show()

    # decode all outputs from the encoder

    # display all decoder outputs (reconstructed images)


if __name__ == "__main__":
    # eval_and_show(autoencoderMLP4Layer)
    # eval_noise_show(autoencoderMLP4Layer)
    # bottleneck_interpolation(autoencoderMLP4Layer)

    parser = argparse.ArgumentParser(description='Evaluate using model')

    # Add arguments for -z, -e, -b, -s, and -p
    parser.add_argument('-l', "--model_weights", type=str, default='MLP.8.pth', help='state dict containing model weights')

    args = parser.parse_args()
    model = autoencoderMLP4Layer()
    device = torch.device("cpu")
    weight_path = f"./data/{args.model_weights}"

    eval_and_show(model, device, weight_path)
    eval_noise_show(model, device, weight_path)
    bottleneck_interpolation(model, device, weight_path)
