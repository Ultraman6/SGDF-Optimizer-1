from torch.optim.optimizer import Optimizer
import torch

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_last_val(self):
        return self.val


class ExpAverage(object):
    def __init__(self, alpha, init_val=0):
        self.val = init_val
        self.avg = init_val
        self.alpha = alpha

    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.avg + (1 - self.alpha) * val

    def get_avg(self):
        return self.avg

    def get_last_val(self):
        return self.val
    
class VanillaKOALA(Optimizer):
    def __init__(
            self,
            params,
            sigma: float = 1,
            q: float = 1,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1,
            **kwargs):
        """
        Implementation of the VanillaKOALA optimizer in PyTorch style

        :param params: parameters to optimize
        :param sigma: initial value of P_k
        :param q: fixed constant Q_k
        :param r: fixed constant R_k (None for online estimation)
        :param alpha_r: smoothing coefficient for online estimation of R_k
        :param weight_decay: weight decay
        :param lr: learning rate
        :param kwargs:
        """
        defaults = dict(sigma=sigma, q=q, r=r, alpha_r=alpha_r, weight_decay=weight_decay, lr=lr, **kwargs)
        super(VanillaKOALA, self).__init__(params, defaults)

        self.eps = 1e-9
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            sigma = group['sigma']
            q = group['q']
            r = group['r']
            alpha_r = group['alpha_r']
            weight_decay = group['weight_decay']
            
            # Initialize state
            #for group in self.param_groups:
            self.state[group]['sigma'] = sigma
            self.state[group]['q'] = q
            if r is not None:
                self.state[group]['r'] = r
            else:
                self.state[group]['r'] = ExpAverage(alpha_r, 1.0)



        #for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.norm(p=2) < self.eps:
                    continue

                layer_grad = d_p + group["weight_decay"] * p.data
                layer_grad_norm = layer_grad.norm(p=2)

                sigma = self.state[group]['sigma']
                q = self.state[group]['q']
                r = self.state[group]['r']

                if isinstance(r, ExpAverage):
                    r.update(layer_grad_norm)
                    cur_r = r.get_avg()
                else:
                    cur_r = r

                s = sigma * (layer_grad_norm ** 2) + cur_r

                layer_loss = loss + 0.5 * group["weight_decay"] * p.data.norm(p=2) ** 2
                scale = group["lr"] * layer_loss * sigma / s
                p.data.add_(-scale * d_p)

                hh_approx = layer_grad_norm ** 2 / s

                sigma -= sigma ** 2 * hh_approx
                self.state[group]['sigma'] = sigma

        return loss
    
class MomentumKOALA(Optimizer):
    def __init__(
            self,
            params,
            sw: float = 1e-1,
            sc: float = 0,
            sv: float = 1e-1,
            a: float = 0.9,
            qw: float = 1e-2,
            qv: float = 1e-2,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1,
            **kwargs):
        """
        Implementation of the MomentumKOALA optimizer in PyTorch style

        :param params: parameters to optimize
        :param sw: initial value of P_k for states
        :param sc: initial value of out of diagonal entries of P_k
        :param sv: initial value of P_k for velocities
        :param a: decay coefficient for velocities
        :param qw: fixed constant Q_k for states
        :param qv: fixed constant Q_k for velocities
        :param r: fixed constant R_k (None for online estimation)
        :param alpha_r: smoothing coefficient for online estimation of R_k
        :param weight_decay: weight decay
        :param lr: learning rate
        :param kwargs:
        """
        defaults = dict(sw=sw, sc=sc, sv=sv, a=a, qw=qw, qv=qv, r=r, alpha_r=alpha_r, weight_decay=weight_decay, lr=lr, **kwargs)
        super(MomentumKOALA, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.eps = 1e-9

        for group in self.param_groups:
            lr = group['lr']  
            sw = group['sw']
            sc = group['sc']
            sv = group['sv']
            a = group['a']
            r = group['r']
            qw = group['qw']
            qv = group['qv']
            alpha_r = group['alpha_r']
            weight_decay = group['weight_decay']
            
            self.shared_device = self.param_groups[0]["params"][0].device
            self.dtype = torch.double

            # Initialize velocities and count params
            self.total_params = 0
            
            # Define state
            self.state["Pt"] = torch.tensor([
                [sw, sc],
                [sc, sv]
            ]).to(self.shared_device).to(self.dtype)

            self.state["qw"] = ExpAverage(0.9, qw)
            self.state["qv"] = qv
            self.state["Q"] = torch.diag(
                torch.tensor([self.state["qw"].get_avg(), self.state["qv"]])
            ).to(self.shared_device).to(self.dtype)

            if r is not None:
                self.state["R"] = r
            else:
                self.state["R"] = ExpAverage(alpha_r, 1.0)

            f = [[1, 1], [0, a]]
            self.state["F"] = torch.tensor(f).to(self.shared_device).to(self.dtype)

            self.state["weight_decay"] = weight_decay
                        
        #for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["vt"] = p.data.new_zeros(p.data.size())
                self.state[p]["gt"] = p.data.new_zeros(p.data.size())
                self.total_params += torch.prod(torch.tensor(list(p.data.size())).to(self.shared_device))

            #for group in self.param_groups:
                #for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.norm(p=2) < self.eps:
                    continue

                layer_grad = d_p + group["weight_decay"] * p.data
                layer_grad_norm = layer_grad.norm(p=2)

                Pt = self.state["Pt"]
                qw = self.state["qw"]
                qv = self.state["qv"]
                r  = self.state["R"]

                if isinstance(r, ExpAverage):
                    r.update(layer_grad_norm)
                    cur_r = r.get_avg()
                else:
                    cur_r = r

                S = layer_grad_norm ** 2 * Pt[0, 0] + cur_r

                layer_loss = loss + 0.5 * group["weight_decay"] * p.data.norm(p=2) ** 2
                    
                K1 = Pt[0, 0] / S * layer_loss * group["lr"]
                K2 = Pt[1, 0] / S * layer_loss * group["lr"]

                # Update weights and velocities
                p.data.sub_(K1 * layer_grad)
                self.state[p]["vt"].sub_(K2 * layer_grad)

                self.state[p]["gt"].mul_(0.9)
                self.state[p]["gt"].add_(0.1 * p.data)

                hh_approx = layer_grad_norm ** 2 / S

                # Update covariance
                HHS = torch.tensor([
                    [hh_approx, 0],
                    [0, 0]
                ]).to(self.shared_device).to(self.dtype)
                PHHS = torch.matmul(Pt, HHS)
                PHHSP = torch.matmul(PHHS, Pt.t())
                Pt.sub_(PHHSP)
            
                    
        return loss
