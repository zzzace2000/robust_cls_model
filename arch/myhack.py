import torch
import captum
from captum.attr import (
    DeepLift,
)


class HackGradAndOutputs(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.normal_grad = None
        self.captum_select_target = None
        self.output = None

    def __enter__(self):
        # Override torch autograd
        if self.is_training:
            self.normal_grad = torch.autograd.grad

            def my_grad(*args, **kwargs):
                kwargs['create_graph'] = True
                kwargs['retain_graph'] = True
                return self.normal_grad(*args, **kwargs)

            torch.autograd.grad = my_grad

        # Override the captum's function to derive the model's output
        self.captum_select_target = \
            captum.attr._utils.common._select_targets
        def my_select_targets(output, target):
            self.output = output
            return self.captum_select_target(output, target)

        captum.attr._utils.common._select_targets = my_select_targets
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_training:
            torch.autograd.grad = self.normal_grad
        captum.attr._utils.common._select_targets = \
            self.captum_select_target

        self.normal_grad = None
        self.captum_select_target = None


class MyDeepLift(DeepLift):
    def __init__(self, *args, kept_backward=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kept_backward = kept_backward
        self.my_kept_module_input = None
        self.my_kept_module_output = None

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
        eps: float = 1e-10,
    ):
        r"""
         Override the function to keep the module input / output
         after back propogation to do 2nd bp.
         """
        if self.kept_backward:
            self.my_kept_module_input = module.input
            self.my_kept_module_output = module.output

        result = super()._backward_hook(module, grad_input, grad_output, eps)

        if self.kept_backward:
            module.input = self.my_kept_module_input
            module.output = self.my_kept_module_output

        return result
