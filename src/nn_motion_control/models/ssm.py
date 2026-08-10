"""
Diagonal state-space model (SSM) layers with a parallel + recurrent dual form.

A state-space layer runs a linear recurrence ``h_t = A h_{t-1} + B u_t`` with a
*diagonal* transition ``A = diag(lambda)`` (S4D/S5 style), read out as
``y_t = Re(C h_t) + D u_t``. Two equivalent evaluations:

  * **parallel** (training on a fixed window): the diagonal recurrence is a first-order
    linear scan, done with a Hillis-Steele associative scan in ``O(log T)`` vectorised
    passes — no serial-over-time loop, so windowed training does not degrade to an RNN;
  * **recurrent** (rollout): one ``h_t = lambda * h_{t-1} + B u_t`` update per step,
    ``O(1)`` in sequence length and carrying *all* history — no receptive-field ceiling.

The two forms are numerically equivalent (to float rounding); ``tests/test_ssm.py``
gates that. The state is complex so a pole pair can model a decaying oscillation;
parameters are stored as real tensors and the pole is built as
``lambda = exp(-exp(log_decay) + i*theta)`` so ``|lambda| < 1`` (stable) by design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (conventional alias)


class DiagSSM(nn.Module):
    """
    One diagonal SSM layer: linear state-space recurrence + skip, GELU, optional res.

    Maps ``[B, T, d_in] -> [B, T, d_out]``. ``forward`` uses the parallel scan; ``step``
    advances one frame from a carried state. Both compute the same per-timestep values.
    """

    def __init__(self, d_in: int, d_out: int, d_state: int):
        super().__init__()
        self.d_in, self.d_out, self.d_state = d_in, d_out, d_state
        self.residual = d_in == d_out
        # Stable complex pole lambda = exp(-exp(log_decay) + i*theta); a spread of
        # timescales (|lambda| ~ 0.98..0.9995 -> effective memory ~50..2000 steps).
        self.log_decay = nn.Parameter(torch.empty(d_state).uniform_(-7.0, -4.0))
        self.theta = nn.Parameter(torch.empty(d_state).uniform_(0.0, 0.31416))
        # B (input -> state) and C (state -> output) are complex; D is a real skip.
        scale_b = d_in**-0.5
        scale_c = d_state**-0.5
        self.b_re = nn.Parameter(torch.randn(d_state, d_in) * scale_b)
        self.b_im = nn.Parameter(torch.randn(d_state, d_in) * scale_b)
        self.c_re = nn.Parameter(torch.randn(d_out, d_state) * scale_c)
        self.c_im = nn.Parameter(torch.randn(d_out, d_state) * scale_c)
        self.d_skip = nn.Parameter(torch.randn(d_out, d_in) * (d_in**-0.5))

    def _lambda(self) -> torch.Tensor:
        mag = torch.exp(-torch.exp(self.log_decay))
        return torch.polar(mag, self.theta)  # complex [d_state]

    def _gamma(self) -> torch.Tensor:
        # Input normalisation gamma = sqrt(1 - |lambda|^2) (the LRU trick): the state
        # accumulates ~1/(1-|lambda|) input contributions, so without this a pole near
        # the unit circle makes the state variance blow up (loss explodes at init). This
        # keeps the state variance ~unit regardless of the pole's memory length.
        mag_sq = torch.exp(-2.0 * torch.exp(self.log_decay))  # |lambda|^2  [d_state]
        return torch.sqrt(1.0 - mag_sq)

    def _kernel(self, t_len: int, device: torch.device) -> torch.Tensor:
        # SSM kernel powers[n, j] = lambda_n^j for j = 0..t_len-1, built from the polar
        # form (no complex ** ) so it is stable and cheap: |lambda|^j decays.
        j = torch.arange(t_len, device=device, dtype=self.log_decay.dtype)
        log_mag = -torch.exp(self.log_decay)  # log|lambda|  [d_state]
        mag_j = torch.exp(log_mag.unsqueeze(1) * j.unsqueeze(0))  # [d_state, t_len]
        ang_j = self.theta.unsqueeze(1) * j.unsqueeze(0)  # [d_state, t_len]
        return torch.polar(mag_j, ang_j)  # [d_state, t_len] complex

    def _bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.complex(self.b_re, self.b_im),  # [d_state, d_in]
            torch.complex(self.c_re, self.c_im),  # [d_out, d_state]
        )

    def _readout(self, y_lin: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Shared nonlinear readout (pointwise in time -> identical in both forms).
        y = F.gelu(y_lin + u @ self.d_skip.t())
        return y + u if self.residual else y

    def scan(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel forward over a window: ``u`` ``[B,T,d_in]`` -> (``y`` ``[B,T,d_out]``,
        final state ``h_last`` ``[B, d_state]`` complex, to prime a streaming rollout).
        """

        b_c, c_c = self._bc()
        bu = torch.einsum("bti,si->bts", u.to(b_c.dtype), b_c)  # [B, T, d_state]
        bu = bu * self._gamma()  # normalise state variance (see _gamma)
        # Parallel state via an FFT convolution of the inputs with the SSM kernel
        # lambda^j (the closed form of the diagonal recurrence). Linear (non-circular)
        # convolution -> pad the transform to >= 2T; native complex, few kernels, so it
        # is far cheaper than an O(log T) scan of full-window complex temporaries.
        t_len = u.shape[1]
        n_fft = 2 * t_len
        kernel = self._kernel(t_len, u.device)  # [d_state, T]
        bu_f = torch.fft.fft(bu, n=n_fft, dim=1)  # [B, n_fft, d_state]
        k_f = torch.fft.fft(kernel, n=n_fft, dim=1).transpose(0, 1)  # [n_fft, d_state]
        h = torch.fft.ifft(bu_f * k_f.unsqueeze(0), dim=1)[:, :t_len]  # [B, T, d_state]
        y_lin = torch.einsum("bts,os->bto", h, c_c).real  # [B, T, d_out]
        return self._readout(y_lin, u), h[:, -1]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.scan(u)[0]

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.d_state, dtype=torch.cfloat, device=device)

    def step(
        self, u_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One recurrent step. ``u_t`` ``[B, d_in]``, state ``h`` ``[B, d_state]`` ->
        (``y_t`` ``[B, d_out]``, new state).
        """

        lam = self._lambda()
        b_c, c_c = self._bc()
        bu = torch.einsum("bi,si->bs", u_t.to(b_c.dtype), b_c) * self._gamma()
        h_new = lam.unsqueeze(0) * h + bu
        y_lin = torch.einsum("bs,os->bo", h_new, c_c).real
        return self._readout(y_lin, u_t), h_new
