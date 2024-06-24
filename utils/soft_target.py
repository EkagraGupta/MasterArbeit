import torch
from torch.nn import functional as F


def soft_target(
    pred,
    gold,
    other=False,
    distribute=True,
    reweight=False,
    soften_one_hot=True,
    lr_correction=False,
):
    # print(gold)
    gold = gold.unsqueeze(1)
    target = gold.long()
    prob = 1 - (gold - target)
    # print()
    weight = torch.clone(prob).float() if reweight else torch.ones_like(prob).float()
    if lr_correction:
        weight = weight / weight.mean()
    n_class = pred.size(1)
    # if we distribute 1-prob to other classes
    scatter_mul = 1.0 if distribute else 0.0
    if soften_one_hot:
        if not other:  # if there is an other class
            one_hot = (
                torch.ones_like(pred) * (1 - prob) * scatter_mul / (n_class - 1)
            ).float()
            one_hot.scatter_(dim=1, index=target, src=prob.float())
        else:
            one_hot = torch.zeros_like(pred)
            one_hot.scatter_(
                dim=1,
                index=torch.ones_like(target) * (n_class - 1),
                src=(1 - prob.float()) * scatter_mul,
            )
            one_hot.scatter_(dim=1, index=target, src=prob.float())
    else:
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(dim=1, index=target, src=torch.ones_like(target).float())

    log_prob = F.log_softmax(pred, dim=1)

    kl = weight * F.kl_div(
        input=log_prob.float(), target=one_hot.float(), reduction="none"
    ).sum(-1)
    return kl.mean()


if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [
                0.1677,
                -0.0024,
                0.1041,
                -0.0754,
                0.0513,
                -0.0438,
                -0.0355,
                0.0365,
                0.0726,
                0.0415,
            ],
        ]
    )
    labels = torch.tensor([0])
    reweight = True
    soften_one_hot = True

    loss = soft_target(
        pred=outputs, gold=labels, reweight=reweight, soften_one_hot=soften_one_hot
    )

    print(loss)
