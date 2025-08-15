import torch
from torchvision import datasets
from torchvision import transforms
import os

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform1 = transforms.Compose([
                    transforms.Resize(config.image_size),
                    # transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor()])
    transform2 = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor()])

    data_root = os.path.join(config.data_dir, config.dataset)
    # svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform1)
    if config.dataset == 'mnist':
      train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform2)
      test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform2)
    elif config.dataset == 'svhn':
      train_set = datasets.SVHN(root = data_root, split = 'train', download = True, transform = transform2)
      test_set = datasets.SVHN(root = data_root, split = 'test', download = True, transform = transform2)
    else:
      train_set = datasets.USPS(root=data_root, train=True, transform=transform1, download=True)
      test_set = datasets.USPS(root=data_root, train=False, transform=transform1, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=config.batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=config.batch_size,
                                              shuffle=False)

    return train_loader, test_loader, UnNormalize(0, -1, identity=True)

class UnNormalize(object):
    def __init__(self, mean, std, identity=False):
        self.mean = mean
        self.std = std
        self.identity = identity

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.identity:
            # Do nothing
            return tensor

        for t, m, s in zip(tensor, self.mean, self.std):
            # The normalize code -> t.sub_(m).div_(s)
            t.mul_(s).add_(m)
        return tensor