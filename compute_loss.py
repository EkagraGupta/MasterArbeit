import torch
from torch.nn import functional as F
import torch.nn as nn


def soft_loss(pred, label, confidence):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target
    label = label.unsqueeze(1)
    confidence = confidence.unsqueeze(1).float()
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    # print(f"\nOnehot: {one_hot}\n")
    one_hot.scatter_(dim=1, index=label, src=confidence)
    # print(f"\nSoftened: {one_hot}\nSum: {torch.sum(one_hot)}")

    # Compute weighted KL loss
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    return kl.mean()


# def soft_loss(pred, label, confidence):
#     label = label.unsqueeze(1)
#     target = label.long()
#     confidence = confidence.unsqueeze(1).float()
#     prob = (1 - (label - target))
#     weight = torch.ones_like(prob).float()
#     n_classes = pred.size(1)

#     one_hot = (torch.ones_like(pred) * (1 - confidence) / (n_classes - 1)).float()
#     one_hot.scatter_(dim=1, index=target, src=confidence)
#     log_prob = F.log_softmax(pred, dim=1)
#     kl = F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none').sum(-1)
#     return kl.mean()


if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [
                -2.0496e-01,
                4.7375e-02,
                -2.7318e-02,
                4.6988e-01,
                1.3611e-01,
                2.1051e-01,
                7.1344e-02,
                1.6107e-01,
                -1.3528e-01,
                -1.5296e-01,
            ],
            [
                -2.4541e-01,
                1.2148e-02,
                -1.3681e-01,
                5.7257e-01,
                1.5858e-01,
                2.2717e-01,
                7.6209e-02,
                8.3832e-02,
                -3.5896e-02,
                -1.9090e-01,
            ],
            [
                -1.4715e-01,
                5.0467e-03,
                -4.3778e-02,
                2.4888e-01,
                1.1313e-01,
                1.2602e-01,
                1.4046e-01,
                3.4285e-02,
                -1.2950e-01,
                2.8288e-02,
            ],
            [
                -2.0420e-01,
                -8.4732e-02,
                -3.0608e-03,
                4.6699e-01,
                8.4846e-02,
                2.1803e-01,
                9.2390e-02,
                1.6728e-01,
                -1.0490e-01,
                -1.0439e-01,
            ],
            [
                -2.0013e-01,
                2.0867e-02,
                -7.0847e-02,
                3.5010e-01,
                1.1229e-01,
                2.0354e-01,
                1.2510e-01,
                8.9791e-02,
                -5.9164e-02,
                -1.1040e-01,
            ],
            [
                -1.5377e-01,
                2.4113e-02,
                -9.9147e-03,
                3.7507e-01,
                5.4192e-02,
                9.9598e-02,
                9.7117e-02,
                7.8155e-02,
                -4.3948e-02,
                -5.0337e-02,
            ],
            [
                -2.2852e-01,
                -1.0367e-01,
                -1.1797e-01,
                5.7465e-01,
                5.5952e-02,
                2.3195e-01,
                1.9876e-01,
                1.9142e-01,
                -6.4853e-02,
                -3.1766e-01,
            ],
            [
                -2.4468e-01,
                -3.4613e-02,
                -4.8807e-02,
                5.6373e-01,
                6.7572e-02,
                1.4073e-01,
                7.0679e-02,
                1.3233e-01,
                -8.0918e-02,
                -3.5420e-01,
            ],
            [
                -2.2395e-01,
                1.8060e-02,
                -8.0375e-02,
                2.9522e-01,
                9.0165e-02,
                1.5558e-01,
                1.1715e-01,
                1.0768e-01,
                -9.5296e-02,
                -2.2295e-02,
            ],
            [
                -3.1785e-01,
                -8.2542e-02,
                -2.1395e-02,
                6.0371e-01,
                1.4389e-01,
                2.1105e-01,
                5.2647e-02,
                2.3914e-01,
                -1.6588e-01,
                -4.5365e-01,
            ],
            [
                -2.4413e-01,
                -2.4940e-02,
                -5.4125e-02,
                5.3818e-01,
                8.3065e-02,
                2.4352e-01,
                1.4110e-01,
                2.8365e-01,
                -8.0776e-02,
                -3.3549e-01,
            ],
            [
                -2.4287e-01,
                -2.2657e-03,
                -5.4868e-02,
                4.8610e-01,
                1.2016e-01,
                1.5458e-01,
                7.0893e-02,
                2.3091e-01,
                -7.3399e-02,
                -1.5045e-01,
            ],
            [
                -1.5868e-01,
                3.3708e-02,
                -4.7587e-02,
                2.2879e-01,
                1.0617e-01,
                1.5082e-01,
                1.7640e-01,
                6.9514e-02,
                -4.1144e-02,
                2.4632e-02,
            ],
            [
                -1.7054e-01,
                1.1915e-02,
                -2.3771e-02,
                3.2384e-01,
                7.9879e-02,
                1.3572e-01,
                1.4904e-01,
                6.9379e-02,
                -1.0727e-01,
                -2.6228e-02,
            ],
            [
                -1.5214e-01,
                -1.3939e-02,
                -5.5676e-02,
                4.6745e-01,
                1.3387e-01,
                9.2954e-02,
                1.2658e-01,
                1.8318e-01,
                -7.2343e-02,
                -8.1601e-02,
            ],
            [
                -4.6442e-01,
                -2.6057e-02,
                -1.9907e-01,
                9.1468e-01,
                6.3309e-02,
                4.1464e-01,
                -1.5870e-01,
                3.6312e-01,
                -7.9865e-02,
                -6.0564e-01,
            ],
            [
                -1.8874e-01,
                -2.3963e-02,
                -5.9693e-02,
                5.5101e-01,
                6.9606e-02,
                2.1900e-01,
                -1.5318e-02,
                1.8902e-01,
                -1.3710e-01,
                -2.3110e-01,
            ],
            [
                -2.9313e-01,
                -9.7980e-02,
                -2.6597e-01,
                7.2215e-01,
                2.0023e-03,
                3.2006e-01,
                2.1294e-05,
                1.7671e-01,
                -1.5324e-01,
                -3.4079e-01,
            ],
            [
                -2.0369e-01,
                -3.8053e-03,
                -8.3162e-02,
                4.2322e-01,
                1.1583e-01,
                1.4724e-01,
                1.0896e-01,
                1.4421e-01,
                -1.1474e-01,
                -9.4047e-02,
            ],
            [
                -2.5782e-01,
                1.7224e-03,
                -3.9400e-02,
                4.4506e-01,
                6.1231e-02,
                2.0577e-01,
                8.4835e-02,
                1.3921e-01,
                -8.7087e-02,
                -6.9793e-02,
            ],
            [
                -2.1683e-01,
                2.5266e-02,
                -3.4570e-02,
                4.5033e-01,
                8.9710e-02,
                1.4898e-01,
                7.5469e-02,
                1.1527e-01,
                -6.9452e-02,
                -1.3445e-01,
            ],
            [
                -2.1163e-01,
                1.4459e-02,
                -1.2258e-01,
                3.8311e-01,
                6.1491e-02,
                1.6602e-01,
                7.5145e-02,
                1.1781e-01,
                -7.0593e-02,
                -1.1165e-01,
            ],
            [
                -3.0794e-01,
                -6.2573e-02,
                -1.9494e-01,
                7.7428e-01,
                6.8949e-02,
                5.1572e-01,
                4.4130e-02,
                3.1149e-01,
                -2.5790e-02,
                -5.4965e-01,
            ],
            [
                -2.9186e-01,
                1.3320e-02,
                -6.5740e-02,
                7.2672e-01,
                1.4231e-01,
                2.6637e-01,
                1.4145e-01,
                2.6548e-01,
                -7.7980e-02,
                -4.7251e-01,
            ],
        ]
    )
    labels = torch.tensor(
        [6, 7, 5, 4, 8, 6, 7, 5, 0, 1, 6, 9, 7, 2, 7, 8, 3, 2, 6, 3, 3, 6, 0, 9]
    )
    confidence = torch.tensor(
        [
            0.6,
            0.7,
            0.5,
            0.4,
            0.8,
            0.6,
            0.7,
            0.5,
            0.0,
            0.1,
            0.6,
            0.9,
            0.7,
            0.2,
            0.7,
            0.8,
            0.3,
            0.2,
            0.6,
            0.3,
            0.3,
            0.6,
            0.0,
            0.9,
        ]
    )
    # confidence = torch.ones(labels.shape[0])
    # print(f'outputs: {outputs.shape}\nlables: {labels.shape}\nconf: {confidence.shape}')

    # soft loss computation
    s_loss = soft_loss(pred=outputs, label=labels, confidence=confidence)

    # conventional loss computation
    criterion = nn.CrossEntropyLoss()
    h_loss = criterion(outputs, labels)
    print(f"s loss: {s_loss}\th loss: {h_loss}")