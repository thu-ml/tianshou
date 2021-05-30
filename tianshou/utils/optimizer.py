import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, Tuple


class AddBias(nn.Module):
    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class SplitBias(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.module(input)
        x = self.add_bias(x)
        return x


class United_Module(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic


class KFACOptimizer(optim.Optimizer):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        lr: float = 0.25,
        momentum: float = 0.9,
        stat_decay: float = 0.99,
        kl_clip: float = 0.001,
        damping: float = 1e-2,
        weight_decay: float = 0.0,
        fast_cnn: bool = False,
        Ts: int = 1,
        Tf: int = 10,
    ) -> None:
        model = United_Module(actor, critic)

        def split_bias(module: nn.Module) -> None:
            for mname, child in module.named_children():
                if hasattr(child, "bias") and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        # rewrite all Linear Module in model. Replace Linear with SplitBias
        split_bias(model)

        super().__init__(model.parameters(), {})

        self.known_modules = {"Linear", "Conv2d", "AddBias"}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa: Dict[nn.Module, torch.Tensor] = {}
        self.m_gg: Dict[nn.Module, torch.Tensor] = {}
        self.Q_a: Dict[nn.Module, torch.Tensor] = {}
        self.Q_g: Dict[nn.Module, torch.Tensor] = {}
        self.d_a: Dict[nn.Module, torch.Tensor] = {}
        self.d_g: Dict[nn.Module, torch.Tensor] = {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

        self.acc_stats = False

    def _save_input(self, module: nn.Module, input: torch.Tensor) -> None:
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info: Tuple[Any, ...] = ()
            if classname == "Conv2d":
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = self.compute_cov_a(input[0].data, classname, layer_info)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            self.m_aa[module] = self.update_running_stat(aa, self.m_aa[module])

    def _save_grad_output(
        self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info: Tuple[Any, ...] = ()
            if classname == "Conv2d":
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = self.compute_cov_g(grad_output[0].data, classname, layer_info)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            self.m_gg[module] = self.update_running_stat(gg, self.m_gg[module])

    def _prepare_model(self) -> None:
        self.modules = []
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in [
                            "Linear", "Conv2d"]) and module.bias is not None), \
                    "You must have a bias as a separate layer"
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self) -> None:
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)  # TODO do not understand

        updates = {}
        for m in self.modules:
            assert len(list(m.parameters())) == 1, \
                "Can handle only one parameter at the moment"
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            p_grad_mat = p.grad.data
            if m.__class__.__name__ == "Conv2d":
                p_grad_mat = p_grad_mat.view(p.grad.data.size(0), -1)

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (
                self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            updates[p] = v.view(p.grad.data.size())

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum().item()
        nu = min(1., (self.kl_clip / vg_sum) ** .5)

        for p in self.model.parameters():
            p.grad.data = updates[p] * nu

        self.optim.step()
        self.steps += 1

    @staticmethod
    def _extract_patches(
        x: torch.Tensor,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...]
    ) -> torch.Tensor:
        if padding[0] + padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        return x.view(*x.shape[:3], -1)

    def compute_cov_a(
        self, a: torch.Tensor, classname: str, layer_info: Tuple[Any, ...]
    ) -> torch.Tensor:
        batch_size = a.size(0)

        if classname == "Conv2d":
            a = KFACOptimizer._extract_patches(a, *layer_info)
            if self.fast_cnn:
                a = a.view(a.size(0), -1, a.size(-1)).mean(1)
            else:
                a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
        elif classname == "AddBias":
            a = torch.ones(a.size(0), 1, device=a.device)

        return a.t() @ (a / batch_size)

    def compute_cov_g(
        self, g: torch.Tensor, classname: str, layer_info: Tuple[Any, ...]
    ) -> torch.Tensor:
        batch_size = g.size(0)

        if classname == "Conv2d":
            if self.fast_cnn:
                g = g.view(g.size(0), g.size(1), -1).sum(-1)
            else:
                g = g.transpose(1, 2).transpose(2, 3).contiguous()
                g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
        elif classname == "AddBias":
            g = g.view(g.size(0), g.size(1), -1).sum(-1)

        g_ = g * batch_size
        return g_.t() @ (g_ / g.size(0))

    def update_running_stat(
        self, aa: torch.Tensor, m_aa: torch.Tensor
    ) -> torch.Tensor:
        return (
            m_aa * self.stat_decay / (1 - self.stat_decay) + aa
        ) * (1 - self.stat_decay)
