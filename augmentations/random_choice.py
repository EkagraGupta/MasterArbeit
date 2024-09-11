import random


class RandomChoiceTransforms:
    def __init__(self, transforms, p, custom=False):
        assert len(transforms) == len(
            p
        ), "The number of transforms and probabilities must match."
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        choice = random.choices(self.transforms, self.p)[0]
        return choice(x)
