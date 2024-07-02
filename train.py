from soft_augment import soft_target
from wideresnet import WideResNet_28_4
from load_augmented_dataset import get_training_dataloader
import time 
import torch.optim as optim
import torch.nn as nn

def compute_loss(outputs, labels, confidence):
    loss = soft_target(pred=outputs,
                       label=labels,
                       confidence=confidence)
    return loss

if __name__=='__main__':
    epochs = 100
    # below from cutout official repo
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    net = WideResNet_28_4(num_classes=10)
    cifar_training_loader = get_training_dataloader(mean=mean,
                                                    std=std,
                                                    batch_size=1,
                                                    da=2,
                                                    aa=1)
    optimizer = optim.SGD(params=net.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=1e-4)
    
    for epoch in range(epochs):
        start = time.time()
        net.train()

        for batch_index, (images, labels, confidences) in enumerate(cifar_training_loader):
            optimizer.zero_grad()
            outputs = net(images)

            loss = compute_loss(outputs=outputs,
                                labels=labels,
                                confidence=confidences)
            loss.backward()
            optimizer.step()

        finish = time.time()
        print(f'Epoch: {epoch}\tLoss: {loss}\tTime: {finish-start}\n')