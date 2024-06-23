from torch import Tensor

def get_softlabel_tensor(label: Tensor) -> Tensor:
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_dict = {}
    for i in range(len(classes)):
        label_dict[classes[i]] = [0. if i!=label else 1.]
    print(label_dict)