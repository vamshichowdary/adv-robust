# %%
import torch
import numpy as np
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
from torchvision import datasets, transforms
import pickle
# %%
ds = CIFAR('/path/to/cifar')
# %%
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, \
                                resume_path='./cifar_l2_0_5.pt')

# %%
use_cuda = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = datasets.CIFAR10(root='/home/vamshi/datasets/CIFAR_10_data/', train=False, download=False, transform=transform_test)# %%

# %%
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=8)
# %%
def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    # Put the model in eval mode so that dropout layers are ineffective
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs, labels
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs, make_adv=False)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return(100 * correct/total)

# %%
acc = calculate_accuracy(model, testloader)
print("acc: ", acc)

# %%
f = open('./resnet50_cifar10_pgd_l2_0_25.p', mode='rb')
ds1 = pickle.load(f)
x_test = ds1["features"]
y_test = ds1["labels"]

# %%
def create_tensor_dataset(X_array, y_array):
    X_torch = torch.from_numpy(X_array)
    y_torch = torch.from_numpy(y_array)
    dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
    return(dataset)

def create_tensor_dataset_with_transform(X_array, y_array, transform):
    X_torch = torch.stack([transform(torch.from_numpy(img)) for img in X_array])
    y_torch = torch.from_numpy(y_array)
    dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
    return(dataset)
# %%
transform_adv = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
advset = create_tensor_dataset_with_transform(x_test, y_test, transform_adv)
# %%
advloader = torch.utils.data.DataLoader(advset, batch_size = 100, num_workers=8)
# %%
adv_acc = calculate_accuracy(model, advloader)
print("adv acc: ", adv_acc)
# %%

# %%

# %%
