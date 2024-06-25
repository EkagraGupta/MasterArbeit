# MasterArbeit

Data augmentation plays a crucial role in machine learning by enhancing the diversity and size of training datasets, which in turn improves model generalization and robustness. Automated augmentation techniques like AutoAugment (AA) and Population Based Augmentation (PBA) have optimized this process by selecting effective augmentation policies. More recent methods, such as TrivialAugment (TA) and RandAugment (RA), offer improved computational efficiency and better results.

This project aims to enhance image classification models through advanced data augmentation strategies that incorporate soft labels. The primary goal is to identify the optimal mapping of soft labels based on specific augmentations. The project will implement a dynamic, adaptive approach similar to Soft Augmentation but tailored to aggressive augmentation policies like TA and RA. This approach aims to increase training efficiency and overall improve model performance.

The project involves several mandatory evaluation criteria, including a thorough understanding of augmentation strategies, implementing image classification benchmark datasets (such as CIFAR-10), and applying label smoothing to generate soft labels based on the severity of the applied augmentations. Additionally, optional evaluation criteria include applying the mapping function to object detection tasks and exploring its adaptation to vision transformers.

The project will be executed through a series of work packages and milestones, including documenting existing augmentation policies, training baseline models, incorporating soft augmentation, and fine-tuning the non-linear mapping function for different augmentation strategies.

The expected outcome is a set of optimized, non-linear mapping functions for soft labels that can be integrated with aggressive augmentation policies, resulting in improved image classification performance.