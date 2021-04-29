import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, List, Type

from tianshou.policy import A2CPolicy
from tianshou.data import Batch


class ACKTRPolicy(A2CPolicy):
    """Implementation of ACKTR
    TODO doc

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        advantage_normalization: bool = True,
        **kwargs: Any,
    ) -> None:
        assert isinstance(optim, KFACOptimizer)
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        del self._grad_norm
        self._norm_adv = advantage_normalization

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(b).dist
                # print(dist.mean[0][0], dist.stddev[0][0])
                # print(self.ret_rms.var)
                if self._norm_adv and False:
                    mean, std = b.adv.mean(), b.adv.std()
                    b.adv = (b.adv - mean) / std  # per-batch norm
                log_prob = dist.log_prob(b.act).reshape(len(b.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * b.adv).mean()
                # calculate loss for critic
                value = self.critic(b.obs).flatten()
                vf_loss = F.mse_loss(b.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                if self.optim.steps % self.optim.Ts == 0:
                    # Compute fisher, see Martens 2014
                    self.optim.model.zero_grad()
                    pg_fisher_loss = -log_prob.mean()

                    value_noise = torch.randn(value.size())
                    if value.is_cuda:
                        value_noise = value_noise.cuda()

                    sample_value = value + value_noise
                    vf_fisher_loss = -(value - sample_value.detach()).pow(2).mean()

                    fisher_loss = pg_fisher_loss + vf_fisher_loss
                    self.optim.acc_stats = True
                    fisher_loss.backward(retain_graph=True)
                    self.optim.acc_stats = False
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class United_Module(nn.Module):
    def __init__(self, actor, critic):
        super(United_Module, self).__init__()
        self.actor = actor
        self.critic = critic


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 actor,
                 critic,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10):
        defaults = dict()
        model = United_Module(actor, critic)

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        # rewrite all Linear Module in model. Replace Linear with SplitBias
        split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

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

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = self.compute_cov_a(input[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            self.update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = self.compute_cov_g(grad_output[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            self.update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in [
                            'Linear', 'Conv2d']) and module.bias is not None), \
                            "You must have a bias as a separate layer"
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)  # TODO do not understand

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (
                self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            # try:
            v = updates[p]
            # except KeyError:
            #     continue
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1., torch.sqrt(self.kl_clip / vg_sum).data)

        for p in self.model.parameters():
            # try:
            v = updates[p]
            # except KeyError:
            #     continue
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1

    @staticmethod
    def _extract_patches(x, kernel_size, stride, padding):
        if padding[0] + padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0), x.size(1), x.size(2),
            x.size(3) * x.size(4) * x.size(5))
        return x

    @staticmethod
    def compute_cov_a(a, classname, layer_info, fast_cnn):
        batch_size = a.size(0)

        if classname == 'Conv2d':
            if fast_cnn:
                a = KFACOptimizer._extract_patches(a, *layer_info)
                a = a.view(a.size(0), -1, a.size(-1))
                a = a.mean(1)
            else:
                a = KFACOptimizer._extract_patches(a, *layer_info)
                a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
        elif classname == 'AddBias':
            is_cuda = a.is_cuda
            a = torch.ones(a.size(0), 1)
            if is_cuda:
                a = a.cuda()

        return a.t() @ (a / batch_size)

    @staticmethod
    def compute_cov_g(g, classname, layer_info, fast_cnn):
        batch_size = g.size(0)

        if classname == 'Conv2d':
            if fast_cnn:
                g = g.view(g.size(0), g.size(1), -1)
                g = g.sum(-1)
            else:
                g = g.transpose(1, 2).transpose(2, 3).contiguous()
                g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
        elif classname == 'AddBias':
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)

        g_ = g * batch_size
        return g_.t() @ (g_ / g.size(0))

    @staticmethod
    def update_running_stat(aa, m_aa, momentum):
        # Do the trick to keep aa unchanged and not create any additional tensors
        m_aa *= momentum / (1 - momentum)
        m_aa += aa
        m_aa *= (1 - momentum)
