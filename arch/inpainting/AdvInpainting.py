import torch
import torch.nn as nn
from advertorch.context import ctx_noparamgrad_and_eval


class AdvInpainting(torch.nn.Module):
    def __init__(self, model, loss_fn=None, eps=0.3, nb_iter=40,
                 alpha=0.375, rand_init=True, clip_min=-1.,
                 clip_max=1., attack='fgsm'):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.alpha = alpha
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.attack = attack

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, X, mask, y):
        if self.attack == 'fgsm':
            with ctx_noparamgrad_and_eval(self.model):
                delta = torch.zeros_like(X).uniform_(
                    -self.eps, self.eps)
                delta.requires_grad = True
                output = self.model(X + delta)
                loss = self.loss_fn(output, y)
                loss.backward()
                grad = delta.grad.detach()
            delta.data = torch.clamp(delta + self.alpha * torch.sign(grad),
                                     -self.eps,
                                     self.eps)
            delta.data = torch.max(torch.min(
                self.clip_max - X, delta.data), self.clip_min - X)
            delta.data[(mask == 1).expand_as(delta)] = 0.
            delta = delta.detach()
        elif self.attack == 'pgd':
            delta = torch.zeros_like(X).uniform_(
                -self.eps, self.eps)
            delta.data = torch.max(torch.min(
                self.clip_max - X, delta.data), self.clip_min - X)
            delta.requires_grad = True

            with ctx_noparamgrad_and_eval(self.model):
                for _ in range(self.nb_iter):
                    delta.data[(mask == 1).expand_as(delta)] = 0.
                    output = self.model(X + delta)
                    loss = self.loss_fn(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(
                        delta + self.alpha * torch.sign(grad),
                        -self.eps,
                        self.eps)[I]
                    delta.data[I] = torch.max(torch.min(
                        self.clip_max - X, delta.data), self.clip_min - X)[I]
                delta.data[(mask == 1).expand_as(delta)] = 0.
                delta = delta.detach()

        return X + delta
