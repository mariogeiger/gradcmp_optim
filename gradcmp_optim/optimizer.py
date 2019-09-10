# pylint: disable=no-member, invalid-name, missing-docstring
import math

import torch


class Optimizer(torch.optim.Optimizer):
    """Implements a continuous version of momentum.

    d/dt velocity = -1/tau (velocity + grad)
     or
    d/dt velocity = -mu/t (velocity + grad)

    d/dt parameters = velocity
    """

    def __init__(self, params, tau, dt=1, low_bound=1e-4, high_bound=1e-3):
        """
        :param tau: momentum parameter, if negative then the characteristic time is t/(-tau)
        """
        defaults = dict(dt=dt, tau=tau, low_bound=low_bound, high_bound=high_bound)
        super().__init__(params, defaults)


    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['t'] = 0
            group['accepted'] = True

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                param_state['grad'] = p.grad.clone()
                param_state['param'] = p.clone()
                param_state['velocity'] = torch.zeros_like(p.data)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            tau = group['tau']

            # init
            if 'step' not in group:
                self.reset()
            else:
                # compute relnorm
                a2, b2, ab = 0, 0, 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]
                    grad = param_state['grad']

                    a2 += grad.norm().pow(2).item()
                    b2 += p.grad.norm().pow(2).item()
                    ab += (grad * p.grad).sum().item()
                relnorm = (a2 + b2 - 2 * ab) / math.sqrt(a2 * b2)

                if relnorm < group['high_bound']:
                    # good
                    group['t'] += group['dt']
                    group['step'] += 1
                    group['accepted'] = True

                    # save current state
                    # save current gradients
                    for p in group['params']:
                        if p.grad is None:
                            continue

                        param_state = self.state[p]
                        param_state['grad'] = p.grad.clone()
                        param_state['param'] = p.clone()
                        param_state['velocity'] = param_state['new_velocity']

                    if relnorm < group['low_bound']:
                        group['dt'] *= 1.1
                else:
                    group['dt'] /= 10
                    group['accepted'] = False

            # propose new state from current
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                dt = group['dt']
                t = group['t']
                g = param_state['grad']
                v = param_state['velocity'].clone()

                if tau > 0:
                    x = math.exp(-dt / tau)
                    v.mul_(x).add_(-(1 - x), g)
                elif tau < 0:
                    mu = -tau
                    x = (t / (t + dt)) ** mu
                    v.mul_(x).add_(-(1 - x), g)
                else:
                    v = -g

                param_state['new_velocity'] = v
                p.data.copy_(param_state['param'] + dt * v)

        return loss
