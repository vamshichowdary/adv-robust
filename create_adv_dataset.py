# %%
import torch
import numpy as np
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
from torchvision import datasets, transforms
import torchvision
import pickle

# %%

BATCH_SIZE = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## import images to be perturbed
#NRM  = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT   = transforms.ToTensor()

## normalization is being done in the robustness model's AttackerModel forward().
#transform_normal = transforms.Compose([TT, NRM])
transform_plain = transforms.Compose([TT])

testset = torchvision.datasets.CIFAR10(root='/home/vamshi/datasets/CIFAR_10_data/', train=False, download=False, transform=transform_plain)
test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle=False, num_workers=8)


# %%
ds = CIFAR('/path/to/cifar')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, \
                                resume_path='./cifar_nat.pt')

model.eval()
model.to(device)
# %%

ATTACK_EPS = 0.25
ATTACK_STEPSIZE = 0.1
ATTACK_STEPS = 20
TARGETED = False

kwargs = {
    'constraint':'2', # use L2-PGD
    'eps': ATTACK_EPS, # L2 radius around original image
    'step_size': ATTACK_STEPSIZE,
    'iterations': ATTACK_STEPS,
    'targeted': TARGETED,
    'do_tqdm': False,
}

# %%
# Loop over all examples in test set
adv_examples = []
adv_labels = []

correct = 0
total = 0
for i, (im, label) in enumerate(test_loader):
    print("processing {}/{}".format(i,len(test_loader)))
    im = im.to(device)
    label = label.to(device)
    pred, im_adv = model(im, label, make_adv=True, **kwargs)
    label_pred = torch.argmax(pred, dim=1)

    total += label.size(0)
    correct += (label_pred == label).sum().item()

    adv_ex = im_adv.detach().cpu().numpy()
    adv_examples.append(adv_ex)
    adv_labels.append(label.detach().cpu().numpy())

print("accuracy on generated adversarial dataset: ",(100 * correct/total))

features = np.concatenate(adv_examples, axis=0)
labels = np.concatenate(adv_labels, axis=0)

dic = {"features":features, "labels":labels}
filehandle = open('./resnet50_cifar10_pgd_l2_0_25.p', 'wb')
pickle.dump(dic, filehandle)
# %%

# %%
