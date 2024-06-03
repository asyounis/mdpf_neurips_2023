import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
from typing import List, Optional

# __all__ = ['RMSprop', 'rmsprop']

class RMSpropCustom(Optimizer):
    r"""Implements RMSprop algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \alpha \text{ (alpha)},\: \gamma \text{ (lr)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm}   \lambda \text{ (weight decay)},\: \mu \text{ (momentum)},\: centered\\
            &\textbf{initialize} : v_0 \leftarrow 0 \text{ (square average)}, \:
                \textbf{b}_0 \leftarrow 0 \text{ (buffer)}, \: g^{ave}_0 \leftarrow 0     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}v_t           \leftarrow   \alpha v_{t-1} + (1 - \alpha) g^2_t
                \hspace{8mm}                                                                     \\
            &\hspace{5mm} \tilde{v_t} \leftarrow v_t                                             \\
            &\hspace{5mm}if \: centered                                                          \\
            &\hspace{10mm} g^{ave}_t \leftarrow g^{ave}_{t-1} \alpha + (1-\alpha) g_t            \\
            &\hspace{10mm} \tilde{v_t} \leftarrow \tilde{v_t} -  \big(g^{ave}_{t} \big)^2        \\
            &\hspace{5mm}if \: \mu > 0                                                           \\
            &\hspace{10mm} \textbf{b}_t\leftarrow \mu \textbf{b}_{t-1} +
                g_t/ \big(\sqrt{\tilde{v_t}} +  \epsilon \big)                                   \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} - \gamma \textbf{b}_t                \\
            &\hspace{5mm} else                                                                   \\
            &\hspace{10mm}\theta_t      \leftarrow   \theta_{t-1} -
                \gamma  g_t/ \big(\sqrt{\tilde{v_t}} + \epsilon \big)  \hspace{3mm}              \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to
    `lecture notes <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ by G. Hinton.
    and centered version `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\gamma/(\sqrt{v} + \epsilon)` where :math:`\gamma`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0,
                 centered=False, foreach: Optional[bool] = None, maximize: bool = False,
                 differentiable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered,
                        weight_decay=weight_decay, foreach=foreach, maximize=maximize,
                        differentiable=differentiable)
        super(RMSpropCustom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            median_stacks = []
            steps = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['median_stacks'] = None
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                square_avgs.append(state['square_avg'])
                median_stacks.append(state['median_stacks'])
                steps.append(state['step'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                if group['differentiable'] and isinstance(state['step'], Tensor):
                    raise RuntimeError('`step` can\'t be a tensor')

                state['step'] += 1


            ms = rmsprop(params_with_grad,
                    grads,
                    square_avgs,
                    median_stacks,
                    steps,
                    grad_avgs,
                    momentum_buffer_list,
                    lr=group['lr'],
                    alpha=group['alpha'],
                    eps=group['eps'],
                    weight_decay=group['weight_decay'],
                    momentum=group['momentum'],
                    centered=group['centered'],
                    foreach=group['foreach'],
                    maximize=group["maximize"],
                    differentiable=group["differentiable"])


            for i, p in enumerate(group['params']):
                state = self.state[p]

                state['median_stacks'] = ms[i]


        return loss


def rmsprop(params: List[Tensor],
            grads: List[Tensor],
            square_avgs: List[Tensor],
            median_stacks: List[Tensor],
            steps: List[int],
            grad_avgs: List[Tensor],
            momentum_buffer_list: List[Tensor],
            # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
            # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
            foreach: bool = None,
            maximize: bool = False,
            differentiable: bool = False,
            *,
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool):
    r"""Functional API that performs rmsprop algorithm computation.
    See :class:`~torch.optim.RMSProp` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        assert(False)
    else:
        func = _single_tensor_rmsprop

    return func(params,
         grads,
         square_avgs,
         median_stacks,
         steps,
         grad_avgs,
         momentum_buffer_list,
         lr=lr,
         alpha=alpha,
         eps=eps,
         weight_decay=weight_decay,
         momentum=momentum,
         centered=centered,
         maximize=maximize,
         differentiable=differentiable)


def _single_tensor_rmsprop(params: List[Tensor],
                           grads: List[Tensor],
                           square_avgs: List[Tensor],
                           median_stacks: List[Tensor],
                           steps: List[int],
                           grad_avgs: List[Tensor],
                           momentum_buffer_list: List[Tensor],
                           *,
                           lr: float,
                           alpha: float,
                           eps: float,
                           weight_decay: float,
                           momentum: float,
                           centered: bool,
                           maximize: bool,
                           differentiable: bool):

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        square_avg = square_avgs[i]
        median_stack = median_stacks[i]
        step = steps[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        is_complex_param = torch.is_complex(param)
        if is_complex_param:
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            square_avg = torch.view_as_real(square_avg)


        g2 = grad**2

        if(median_stack is None):
            median_stack = g2[...].unsqueeze(0)
        else:
            median_stack = torch.cat([median_stack, g2[...].unsqueeze(0)], dim=0)

            if(median_stack.shape[0] > 100):
                cut = median_stack.shape[0] - 100
                median_stack = median_stack[cut:]

        median_stacks[i] = median_stack


        if(median_stack.shape[0] > 100):
            median,_ = torch.median(median_stack, dim=0)
            avg = median.sqrt()
            avg = avg.add_(eps)
        else:
            avg = torch.ones_like(grad)


        # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        # avg = square_avg.sqrt()
        # avg = avg.add_(eps)


        if momentum > 0:
            buf = momentum_buffer_list[i]
            if is_complex_param:
                buf = torch.view_as_real(buf)
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)



    return median_stacks