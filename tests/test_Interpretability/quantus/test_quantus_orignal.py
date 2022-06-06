import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

# Imports general.
import sys
import gc
import warnings
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import csmt.Interpretability.captum
from csmt.Interpretability.captum.attr import *
import random
import os
import cv2

import csmt.Interpretability.quantus as quantus


# Notebook settings.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
warnings.filterwarnings("ignore", category=UserWarning)

# Load datasets and make loaders.
test_samples = 24
transformer = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='csmt/datasets/data/mnist/', train=True, transform=transformer, download=True)
test_set = torchvision.datasets.MNIST(root='csmt/datasets/data/mnist/', train=False, transform=transformer, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True) # num_workers=4,
test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, pin_memory=True)

# Load a batch of inputs and outputs to use for evaluation.
x_batch, y_batch = iter(test_loader).next()
x_batch, y_batch = x_batch.to(device), y_batch.to(device)

# Plot some inputs!
# nr_images = 5
# fig, axes = plt.subplots(nrows=1, ncols=nr_images, figsize=(nr_images*3, int(nr_images*2/3)))
# for i in range(nr_images):
#     axes[i].imshow((np.reshape(x_batch[i].cpu().numpy(), (28, 28)) * 255).astype(np.uint8), vmin=0.0, vmax=1.0, cmap="gray")
#     axes[i].title.set_text(f"MNIST class - {y_batch[i].item()}")
#     axes[i].axis("off")
# plt.show()

class LeNet(torch.nn.Module):
    """Network architecture from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch."""
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 6, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(256, 120)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(120, 84)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.fc_3(x)
        return x

# Load model architecture.
model = LeNet()
print(f"\n Model architecture: {model.eval()}\n")

def train_model(model, 
                train_data: torchvision.datasets,
                test_data: torchvision.datasets, 
                device: torch.device, 
                epochs: int = 20,
                criterion: torch.nn = torch.nn.CrossEntropyLoss(), 
                optimizer: torch.optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9), 
                evaluate: bool = False):
    """Train torch model."""
    
    model.train()
    
    for epoch in range(epochs):

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model!
        if evaluate:
            predictions, labels = evaluate_model(model, test_data, device)
            test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())
        
        print(f"Epoch {epoch+1}/{epochs} - test accuracy: {(100 * test_acc):.2f}% and CE loss {loss.item():.2f}")

    return model

def evaluate_model(model, data, device):
    """Evaluate torch model."""
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor().to(device)

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            logits = torch.cat([logits, model(images)])
            targets = torch.cat([targets, labels])
    
    return torch.nn.functional.softmax(logits, dim=1), targets

path_model_weights='tests/Interpretability/quantus/model.pt'
model.load_state_dict(torch.load(path_model_weights))
# Train and evaluate model.
# model = train_model(model=model.to(device),
#                     train_data=train_loader,
#                     test_data=test_loader,
#                     device=device,
#                     epochs=20,
#                     criterion=torch.nn.CrossEntropyLoss().to(device),
#                     optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
#                     evaluate=True)
# torch.save(model.state_dict(), path_model_weights)

model.to(device)
model.eval()

# Check test set performance.
predictions, labels = evaluate_model(model, test_loader, device)
test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())        
print(f"Model test accuracy: {(100 * test_acc):.2f}%")

# print(x_batch.shape)
# print(y_batch.shape)
# print(x_batch.dtype)
# print(y_batch.dtype)
# Generate normalised Saliency and Integrated Gradients attributions of the first batch of the test set.
a_batch_saliency = quantus.normalise_by_negative(Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
a_batch_intgrad = quantus.normalise_by_negative(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch, baselines=torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy())

# Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

# Plot explanations!
nr_images = 3
fig, axes = plt.subplots(nrows=nr_images, ncols=3, figsize=(nr_images*2.5, int(nr_images*3)))
for i in range(nr_images):
    axes[i, 0].imshow((np.reshape(x_batch[i], (28, 28)) * 255).astype(np.uint8), vmin=0.0, vmax=1.0, cmap="gray")
    axes[i, 0].title.set_text(f"MNIST digit {y_batch[i].item()}")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(a_batch_saliency[i], cmap="seismic")
    axes[i, 1].title.set_text(f"Saliency")
    axes[i, 1].axis("off")
    a = axes[i, 2].imshow(a_batch_intgrad[i], cmap="seismic")
    axes[i, 2].title.set_text(f"Integrated Gradients")
    axes[i, 2].axis("off")
# plt.savefig(f'{path}/quantus/tutorials/assets/mnist_example.png', dpi=400)
plt.tight_layout()
plt.show()

# Define params for evaluation.
params_eval = {
    "nr_samples": 10,
    "perturb_radius": 0.1,
    "norm_numerator": quantus.fro_norm,
    "norm_denominator": quantus.fro_norm,
    "perturb_func": quantus.uniform_noise,
    "similarity_func": quantus.difference,
    "disable_warnings": True,
}

# Return max sensitivity scores in an one-liner - by calling the metric instance.
# scores_saliency = quantus.MaxSensitivity(**params_eval)(model=model, 
#    x_batch=x_batch,
#    y_batch=y_batch,
#    a_batch=a_batch_saliency,
#    **{"explain_func": quantus.explain, "method": "Saliency", "device": device, "img_size": 28, "normalise": False, "abs": False})

# # Return max sensitivity scores in an one-liner - by calling the metric instance.
# scores_intgrad = quantus.MaxSensitivity(**params_eval)(model=model, 
#    x_batch=x_batch,
#    y_batch=y_batch,
#    a_batch=a_batch_intgrad,
#    **{"explain_func": quantus.explain, "method": "IntegratedGradients", "device": device, "img_size": 28, "normalise": False, "abs": False})

# print(f"max-Sensitivity scores by Yeh et al., 2019\n" \
#       f"\n • Saliency = {np.mean(scores_saliency):.2f} ({np.std(scores_saliency):.2f})." \
#       f"\n • Integrated Gradients = {np.mean(scores_intgrad):.2f} ({np.std(scores_intgrad):.2f})."
#       )

# metrics = {"max-Sensitivity": quantus.MaxSensitivity(**params_eval)}

metrics = {"max-Sensitivity": quantus.MaxSensitivity(**params_eval)}

xai_methods = {"Saliency": a_batch_saliency,
               "IntegratedGradients": a_batch_intgrad}

results = quantus.evaluate(metrics=metrics,
                           xai_methods=xai_methods,
                           model=model,
                           x_batch=x_batch,
                           y_batch=y_batch,
                           agg_func=np.mean,
                           **{"explain_func": quantus.explain, "device": device, "img_size": 28, "normalise": False, "abs": False})

df = pd.DataFrame(results)
print(df)

