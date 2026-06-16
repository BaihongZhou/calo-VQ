import sys
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from calo_ldm.util import (
    instantiate_from_config,
    hinge_d_loss, vanilla_d_loss,
    adopt_weight, measure_perplexity,
    load_geom_mask,
)


class CombinedLoss(pl.LightningModule):
    """Stage-1 multi-term loss for the DarkSHINE xyz VQ-VAE.

    Terms: masked reconstruction (L1/L2 over real crystals only), codebook +
    commitment (passed in via pred['qloss']), shower energy-centre and width
    regularisers computed in (x, y) per depth-layer, and the adversarial GAN
    term with the taming-transformers adaptive weight. The mask makes every
    pixel-space sum run over real crystals only -- padding cells never produce a
    gradient and never enter a normalisation denominator.
    """
    def __init__(self, disc_start,
                 codebook_weight=1.0,
                 disc_factor=1.0, disc_weight=1.0, disc_loss="hinge",
                 n_embed=None,
                 pixel_power=2, pixel_weight=1,
                 ec_weight=0., width_weight=0.,
                 hit_weight=0., hit_threshold=0.025, hit_pos_weight_max=50.,
                 fp_weight=0., fp_threshold=None, fp_temperature=None, fp_start=0,
                 disc_config=None,
                 cond_dim=0, log_scale_params=None,
                 reco_normalization='R', disc_normalization='U',
                 mask_path='DarkSHINE_data/geom_mask.npy',
                 dataset_name='darkshine',
                 cell_pitch=1.25,            # half-crystal grid pitch [cm]
                 adaptive_max=1e4, adaptive_min=0.0,
                 **unused):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_power = pixel_power
        self.pixel_weight = pixel_weight
        self.ec_weight = ec_weight
        self.width_weight = width_weight
        # hit/no-hit (occupancy) head: masked, class-imbalance-weighted BCE on
        # (E_cell > hit_threshold). hit_threshold is per-43x43-cell (= crystal 0.1 MeV
        # / 4, since a hit crystal splits equally over its 2x2 block in truth).
        self.hit_weight = hit_weight
        self.hit_threshold = hit_threshold
        self.hit_pos_weight_max = hit_pos_weight_max
        # soft-occupancy false-positive penalty applied DIRECTLY to the energy
        # output (not a detached head): a differentiable soft gate
        # sigmoid((E_pred - tau)/T) penalised only where truth is dark, so the
        # energy decoder itself learns to keep truth-dark cells below threshold.
        # tau shares the hit grid (per-43x43-cell MeV); T defaults to tau/5 so a
        # ~0 cell reads sigmoid(-5)~=0.007. fp_start ramps it in (NaN safety).
        self.fp_weight = fp_weight
        self.fp_threshold = hit_threshold if fp_threshold is None else fp_threshold
        self.fp_temperature = (self.fp_threshold / 5.0) if fp_temperature is None \
            else fp_temperature
        self.fp_start = fp_start
        self.width_eps = 1e-2       # [cm^2] floor inside sqrt -> bounds the width gradient
        self.reco_normalization = reco_normalization
        self.disc_normalization = disc_normalization
        self.adaptive_max = adaptive_max
        self.adaptive_min = adaptive_min
        self.n_embed = n_embed
        self.dataset_name = dataset_name

        self.discriminator = instantiate_from_config(
            disc_config, {'cond_dim': cond_dim, 'log_scale_params': log_scale_params}, overwrite=False)
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"CombinedLoss (xyz) running with {disc_loss} loss.", file=sys.stderr)

        # fixed geometry mask (depth, x, y) = (11, 43, 43). These are constants
        # derived from mask_path/cell_pitch, so persistent=False keeps them out of
        # the checkpoint (and avoids loading non-contiguous buffers).
        mask = load_geom_mask(mask_path).float()
        self.register_buffer('mask', mask[None].contiguous(), persistent=False)   # (1, 11, 43, 43)

        # (x, y) cell coordinates for shower centre / width, broadcast over depth.
        n = mask.shape[-1]                                     # 43
        coord = torch.arange(n).float() * cell_pitch
        x_grid = coord[:, None].expand(n, n).contiguous()     # varies along x (-2)
        y_grid = coord[None, :].expand(n, n).contiguous()     # varies along y (-1)
        self.register_buffer('x_grid', x_grid[None, None].contiguous(), persistent=False)   # (1,1,43,43)
        self.register_buffer('y_grid', y_grid[None, None].contiguous(), persistent=False)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, self.adaptive_min, self.adaptive_max).detach()
        return d_weight * self.discriminator_weight

    def _centre_width(self, energy):
        # energy ~ (N, depth, x, y); returns per-(N,depth) centre & width in x and y.
        # Floor the per-layer denominator at a small fraction of the shower's total
        # energy: an empty/near-empty depth layer has an ill-defined centroid, and a
        # ~1e-16 floor lets 1/s ~ 1e16 blow up the gradient (this -- not the sqrt --
        # is what wrecks nll_loss). The floor is detached so it only bounds 1/s.
        s_layer = energy.sum(axis=(-1, -2))                   # (N, depth)
        floor = (1e-3 * s_layer.sum(-1, keepdim=True)).detach().clamp(min=1e-3)
        s = torch.maximum(s_layer, floor)
        ec_x = (self.x_grid * energy).sum(axis=(-1, -2)) / s
        ec_y = (self.y_grid * energy).sum(axis=(-1, -2)) / s
        x2 = (self.x_grid ** 2 * energy).sum(axis=(-1, -2)) / s
        y2 = (self.y_grid ** 2 * energy).sum(axis=(-1, -2)) / s
        # eps inside the sqrt: d/dv sqrt(v) -> inf as v->0, so a depth layer whose
        # (predicted) energy collapses toward a single cell (variance ~0, common in
        # deep layers / a sharpened softmax) produces a NaN/exploding gradient that
        # blows up nll_loss. eps bounds the gradient to 1/(2*sqrt(eps)).
        wx = torch.sqrt((x2 - ec_x ** 2).clip(min=0.) + self.width_eps)
        wy = torch.sqrt((y2 - ec_y ** 2).clip(min=0.) + self.width_eps)
        return ec_x, ec_y, wx, wy

    def forward(self, batch, pred, optimizer_idx, global_step, last_layer=None, split='train'):
        codebook_loss = pred['qloss']
        mask = self.mask

        # ---- masked reconstruction loss (over real crystals only) ----
        reco_true = batch[f'pixels_{self.reco_normalization}']
        reco_pred = pred[f'pixels_{self.reco_normalization}_pred']
        diff = (reco_true - reco_pred) * mask
        l1_loss = diff.abs().sum(axis=(-1, -2, -3)).mean()
        l2_loss = diff.pow(2).sum(axis=(-1, -2, -3)).mean()
        rec_loss = l1_loss if self.pixel_power == 1 else l2_loss

        # ---- shower centre / width in (x, y) per depth-layer ----
        e_true = batch['pixels_E_orig']
        e_pred = pred['pixels_E_pred']
        ecx_t, ecy_t, wx_t, wy_t = self._centre_width(e_true)
        ecx_p, ecy_p, wx_p, wy_p = self._centre_width(e_pred)
        ec_loss = 0.5 * ((ecx_t - ecx_p) ** 2 + (ecy_t - ecy_p) ** 2).mean()
        width_loss = 0.5 * ((wx_t - wx_p) ** 2 + (wy_t - wy_p) ** 2).mean()

        # ---- hit/no-hit (occupancy) head ----
        hit_loss = torch.zeros((), device=reco_true.device)
        hit_occ = torch.zeros((), device=reco_true.device)
        if self.hit_weight > 0 and 'pixels_hit_logits' in pred:
            hit_logits = pred['pixels_hit_logits']
            target = (e_true > self.hit_threshold).to(hit_logits.dtype)   # (N,11,43,43)
            n = target.shape[0]
            denom = mask.sum() * n
            with torch.no_grad():
                pos = (target * mask).sum()
                pos_weight = ((denom - pos) / pos.clamp(min=1.0)).clamp(1.0, self.hit_pos_weight_max)
            bce = F.binary_cross_entropy_with_logits(
                hit_logits, target, pos_weight=pos_weight, reduction='none')
            hit_loss = (bce * mask).sum() / denom.clamp(min=1.0)
            hit_occ = ((torch.sigmoid(hit_logits) > 0.5).to(target.dtype) * mask).sum() / n

        # ---- soft occupancy: one-sided false-positive penalty on the energy ----
        # Acts on e_pred directly (gradient -> energy decoder), unlike the detached
        # hit head. Only penalises lighting truth-dark cells; recon handles the
        # bright ones, so this adds sparsity pressure with minimal fight vs width.
        fp_loss = torch.zeros((), device=reco_true.device)
        if self.fp_weight > 0:
            n_fp = e_true.shape[0]
            p_hit = torch.sigmoid((e_pred - self.fp_threshold) / self.fp_temperature)
            dark = (e_true <= self.fp_threshold).to(p_hit.dtype)
            denom_fp = (mask.sum() * n_fp).clamp(min=1.0)
            fp_loss = (p_hit * dark * mask).sum() / denom_fp
            fp_loss = adopt_weight(1.0, global_step, threshold=self.fp_start) * fp_loss

        nll_loss = self.pixel_weight * rec_loss
        if self.ec_weight > 0:
            nll_loss = nll_loss + self.ec_weight * ec_loss
        if self.width_weight > 0:
            nll_loss = nll_loss + self.width_weight * width_loss
        if self.hit_weight > 0:
            nll_loss = nll_loss + self.hit_weight * hit_loss
        if self.fp_weight > 0:
            nll_loss = nll_loss + self.fp_weight * fp_loss

        disc_true = batch[f'pixels_{self.disc_normalization}']
        disc_pred = pred[f'pixels_{self.disc_normalization}_pred']

        if optimizer_idx == 0:
            # generator update
            if self.discriminator_weight > 0:
                logits_fake = self.discriminator(disc_pred.contiguous(), batch['cond'])
                g_loss = -torch.mean(logits_fake)
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0, device=disc_true.device)
                disc_factor = torch.tensor(
                    adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start),
                    device=disc_true.device)
            else:
                g_loss = disc_factor = d_weight = torch.tensor(0.0, device=disc_true.device)

            assert len(codebook_loss.shape) <= 1
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss

            log = {f"{split}/total_loss": loss.detach().clone(),
                   f"{split}/quant_loss": codebook_loss.detach().clone(),
                   f"{split}/nll_loss": nll_loss.detach().clone(),
                   f"{split}/l1_loss": l1_loss.detach().clone(),
                   f"{split}/l2_loss": l2_loss.detach().clone(),
                   f"{split}/ec_loss": ec_loss.detach().clone(),
                   f"{split}/width_loss": width_loss.detach().clone(),
                   f"{split}/rec_loss": rec_loss.detach().clone(),
                   f"{split}/hit_loss": hit_loss.detach().clone(),
                   f"{split}/hit_occ": hit_occ.detach().clone(),
                   f"{split}/fp_loss": fp_loss.detach().clone(),
                   f"{split}/d_weight": d_weight.detach().clone(),
                   f"{split}/disc_factor": disc_factor.detach().clone(),
                   f"{split}/g_loss": g_loss.detach().clone()}
            with torch.no_grad():
                perplexity, cluster_usage = measure_perplexity(pred['indices'], self.n_embed)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
                log[f"{split}/L1_ec_x"] = torch.abs(ecx_p - ecx_t).mean()
                log[f"{split}/L1_ec_y"] = torch.abs(ecy_p - ecy_t).mean()
                log[f"{split}/L1_width_x"] = torch.abs(wx_p - wx_t).mean()
                log[f"{split}/L1_width_y"] = torch.abs(wy_p - wy_t).mean()
            return loss, log

        if optimizer_idx == 1:
            # discriminator update
            logits_real = self.discriminator(disc_true.contiguous().detach(), batch['cond'])
            logits_fake = self.discriminator(disc_pred.contiguous().detach(), batch['cond'])
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {f"{split}/disc_loss": d_loss.detach().clone(),
                   f"{split}/logits_real": logits_real.detach().mean(),
                   f"{split}/logits_fake": logits_fake.detach().mean()}
            return d_loss, log
