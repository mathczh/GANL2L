import torch
from torch.optim.optimizer import Optimizer, required

class ProjSGD(Optimizer):
    r"""
    Do projection after sgd.
    """

    def __init__(self, model, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, reg_weight=1e-5,
                 project_norm = False, orth_reg = False, l2_reg = False):
        """
        project_norm, orth_reg, l2_reg are for the specialconvLayer
        """
        names = [name for name, param in model.named_parameters()]
        params = model.parameters()
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, reg_weight=reg_weight, names=names)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ProjSGD, self).__init__(params, defaults)
        self.project_norm = project_norm
        self.orth_reg = orth_reg
        self.l2_reg = l2_reg
        self.model = model

    def __setstate__(self, state):
        super(ProjSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print('######### Step Separator############')
        # use_cuda = torch.cuda.is_available()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            reg_weight = group['reg_weight']
            names = group['names']

            specialconvLayer = [i for i in range(len(names)) if ('sconv' in names[i])]

            # print('######### Group Separator############')

            for i in range(len(group['params'])):

                p = group['params'][i]

                if p.grad is None:
                    continue

                if i in specialconvLayer:
                    d_p = p.grad.data
                    if self.orth_reg or self.manifold_grad:
                        originSize = p.data.size()
                        outputSize = originSize[0]
                        W = p.data.view(outputSize, -1)
                        Wt = torch.t(W)
                        WWt = W.mm(Wt)

                    if self.orth_reg:
                        # gradW = Original d_p
                        gradW = d_p.view(outputSize, -1)
                        # shift = lambda * (WWt-I)W
                        I = torch.eye(WWt.size()[0])
                        shift = (WWt.sub(I)).mm(W)

                        d_p = gradW + reg_weight * shift
                        d_p = d_p.view(originSize)

                    elif self.l2_reg:
                        if weight_decay != 0:
                            d_p.add_(weight_decay, p.data)

                else: # for others
                    d_p = p.grad.data
                    # add weight decay
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                # adjust with momentum
                # veloc = mom * veloc + grad
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # W = W - lr * veloc
                p.data.add_(-group['lr'], d_p)

            # W_conv = W_conv / (W_conv.norm)
            if self.project_norm:
                    self.model.project()

        return loss
