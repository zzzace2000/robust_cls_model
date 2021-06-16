import torch


class NoiseDatasetBase(object):
    '''
    Generate a Gaussian noise with mean 0.5 and stdev 1,
    clip between 0 and 1, and normalize to -1 and 1.
    '''
    def __init__(self, num_samples, num_channels=3, seed=42):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.seed = seed
        self.generator = torch.Generator()

    def make_loader(self, batch_size, **kwargs):
        # Generate the same data each time
        self.generator.manual_seed(self.seed)

        num_times = (self.num_samples // batch_size)
        for _ in range(num_times):
            n = self.generate_noise(batch_size)
            yield n, 0

        residual = self.num_samples - num_times * batch_size
        if residual > 0:
            n = self.generate_noise(residual)
            yield n, 0

    def generate_noise(self, num):
        raise NotImplementedError()

    def __len__(self):
        return self.num_samples


class GaussianNoiseDataset(NoiseDatasetBase):
    '''
    Generate a Gaussian noise with mean 0.5 and stdev 1,
    clip between 0 and 1, and normalize to -1 and 1.
    '''
    def generate_noise(self, num):
        gn = torch.randn(num, self.num_channels, 224, 224, generator=self.generator).add_(0.5)
        gn.clamp_(min=0., max=1.).sub_(0.5).mul_(2)
        return gn


class UniformNoiseDataset(NoiseDatasetBase):
    '''
    Generate a Gaussian noise with mean 0.5 and stdev 1,
    clip between 0 and 1, and normalize to -1 and 1.
    '''
    def generate_noise(self, num):
        un = torch.rand(num, self.num_channels, 224, 224, generator=self.generator).sub_(0.5).mul_(2)
        return un
