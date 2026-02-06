import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v: torch.Tensor, t: torch.Tensor, x_shape):

    out = torch.gather(v, dim=0, index=t).float().to(t.device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    
    def __init__(self, model: nn.Module, beta_1: float, beta_T: float, T: int):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0: torch.Tensor, cond: torch.Tensor):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        eps = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps
        pred = self.model(x_t, t, cond)  # eps prediction
        return F.mse_loss(pred, eps, reduction="none")



# Posterior-Annealed Conditional Diffusion Sampler
# ---------------------------

class IdentityDegradation(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GaussianDiffusionPACDSampler(nn.Module):
    
    def __init__(
        self,
        model: nn.Module,
        beta_1: float,
        beta_T: float,
        T: int,
        w_cfg: float = 0.0,              
        alpha_max: float = 0.9,            
        sigma_c: float = 1.0,              
        degradation: nn.Module | None = None,
    ):
        super().__init__()
        self.model = model
        self.T = T
        self.w_cfg = w_cfg
        self.Ts = Ts
        self.alpha_max = alpha_max

        self.register_buffer("inv_sigma_c2", torch.tensor(1.0 / (sigma_c ** 2), dtype=torch.float32))

        self.D = degradation if degradation is not None else IdentityDegradation()

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1.0)[:T]

        self.register_buffer("coeff1", torch.sqrt(1.0 / alphas))
        self.register_buffer("coeff2", self.coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar))
        self.register_buffer("posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))


    def _predict_xt_prev_mean_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):

        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def _p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
      
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps_c = self.model(x_t, t, cond)
        if self.w_cfg != 0.0:
            eps_u = self.model(x_t, t, torch.zeros_like(cond).to(cond.device))
            eps = (1.0 + self.w_cfg) * eps_c - self.w_cfg * eps_u
        else:
            eps = eps_c

        mean = self._predict_xt_prev_mean_from_eps(x_t, t, eps)
        return mean, var

    # ---- posterior annealing schedule α(t) ----
    def _alpha_t(self, time_step: int) -> float:

        threshold = max(self.T - self.Ts, 0)
        if time_step >= threshold:
            span = max(self.T - 1 - threshold, 1)
            return float(self.alpha_max * (time_step - threshold) / span)
        return 0.0

    def _likelihood_grad(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
 
        x_req = x.detach().requires_grad_(True)
        Dx = self.D(x_req)
        
        loss = (Dx - c).pow(2).mean() * self.inv_sigma_c2
        grad = torch.autograd.grad(loss, x_req, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    @torch.no_grad()
    def forward(
        self,
        x_T: torch.Tensor,
        cond: torch.Tensor,
        start_t: int | None = None,
        obs: torch.Tensor | None = None,
    ):
       
        x_t = x_T
        if start_t is None:
            start_t = self.T - 1
        for time_step in reversed(range(start_t + 1)):
            t = x_t.new_full((x_T.shape[0],), time_step, dtype=torch.long)

            mean, var = self._p_mean_variance(x_t=x_t, t=t, cond=cond)
            noise = torch.randn_like(x_t) if time_step > 0 else 0.0
            x_next = mean + torch.sqrt(var) * noise
            if obs is not None:
                alpha = self._alpha_t(time_step)
                if alpha > 0:
                    
                    eta_t = torch.sqrt(var).float()
                    grad = self._likelihood_grad(x_next, obs)
                    x_next = x_next - eta_t * (alpha * grad)

            x_t = x_next
            if torch.isnan(x_t).any():
                raise RuntimeError("NaN encountered during sampling.")

        return torch.clamp(x_t, -1.0, 1.0)
