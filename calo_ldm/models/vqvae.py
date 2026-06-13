import sys
from contextlib import contextmanager

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from calo_ldm.layers import VectorQuantizer
from calo_ldm.util import instantiate_from_config, load_geom_mask
from calo_ldm.ema import LitEma
from calo_ldm.metrics import ShowerMetrics
from calo_ldm.geometry import downsample_to_crystals


class VQModel(pl.LightningModule):
    """Stage-1 VQ-VAE for the DarkSHINE xyz calorimeter.

    Geometry mapping (see CLAUDE.md): the data is a channels-last regular grid
    (N, x=43, y=43, depth=11). The dataloader permutes it to channels-first
    (N, depth=11, x=43, y=43); depth (shower evolution, no translation symmetry)
    is the CHANNEL axis -- analogous to the radial axis in the paper -- while the
    (x, y) plane (translation-symmetric) is the 2D conv image.

    Pipeline: encode -> vector-quantize (codebook) -> decode (masked softmax).
    The decoder output sums to 1 over the real crystals only; the per-shower
    energy ratio R = E_deposited / E_incident carries the scale and is generated
    by the stage-2 GPT. Trained with reconstruction + codebook/commitment +
    adversarial (GAN discriminator) losses. Two optimizers are driven via PL2
    manual optimization.
    """
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 loss_config,
                 n_embed,
                 embed_dim,
                 cond_dim=0,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 use_ema=False,
                 mask_path='DarkSHINE_data/geom_mask.npy',
                 dataset_name='darkshine',
                 log_scale_params=(0, 3e3, 7),
                 reco_normalization='R',   # compare pixels in R (sum-to-R) space
                 disc_normalization='U',   # discriminator sees U (sum-to-1) space
                 sane_index_shape=True,
                 do_metric=True,           # log DarkSHINE reco-vs-truth physics metrics
                 metric_freq=1,            # emit histogram/figure metrics every N val epochs
                 metric_hit_threshold=0.1, # [MeV] cell energy above which a cell is a "hit"
                 downsample_mode='sum',    # 43x43 -> 21x21 crystal aggregation: 'sum' or 'min'
                 convert_to_detector_shape=False,  # if True, postprocess output is (N,11,21,21)
                 hit_gate=True,            # apply the decoder hit/no-hit gate at eval/generation
                 hit_gate_threshold=0.5,   # sigmoid(hit_logits) > thr -> cell kept
                 freeze_except_hit_head=False,  # train ONLY the decoder hit head (energy model frozen)
                 **unused):                # absorb any leftover config keys
        super().__init__()
        # PL2: two optimizers (AE + discriminator) -> manual optimization.
        self.automatic_optimization = False

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.dataset_name = dataset_name

        self.reco_normalization = reco_normalization
        self.disc_normalization = disc_normalization

        self.do_metric = do_metric
        self.metric_freq = metric_freq
        self.metric_hit_threshold = metric_hit_threshold
        self.downsample_mode = downsample_mode
        self.convert_to_detector_shape = convert_to_detector_shape
        self.hit_gate = hit_gate
        self.hit_gate_threshold = hit_gate_threshold
        self.freeze_except_hit_head = freeze_except_hit_head
        self._val_metrics = None

        # fixed geometry mask (depth, x, y) = (11, 43, 43); same for every shower.
        # persistent=False: constant derived from mask_path, kept out of the checkpoint.
        self.register_buffer('mask', load_geom_mask(mask_path).contiguous(), persistent=False)

        print("Overwrite mode: shared params (cond_dim/log_scale_params/mask_path) are "
              "pushed into the encoder/decoder/loss sub-configs.", file=sys.stderr)
        self.encoder = instantiate_from_config(encoder_config, {
            'cond_dim': cond_dim, 'log_scale_params': log_scale_params}, overwrite=True)
        self.decoder = instantiate_from_config(decoder_config, {
            'cond_dim': cond_dim, 'mask_path': mask_path}, overwrite=True)
        self.loss = instantiate_from_config(loss_config, {
            'cond_dim': cond_dim, 'n_embed': n_embed,
            'log_scale_params': log_scale_params,
            'disc_normalization': disc_normalization,
            'reco_normalization': reco_normalization,
            'mask_path': mask_path,
            'dataset_name': dataset_name}, overwrite=True)

        assert sane_index_shape
        # pixels_dim=3: latent is a 2D image (h, w) with `embed_dim` channels.
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        sane_index_shape=sane_index_shape, pixels_dim=3)
        self.quant_conv = torch.nn.Conv2d(self.encoder.ch_out, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.encoder.ch_out, 1)

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if self.freeze_except_hit_head:
            # Train ONLY the decoder hit head on top of a frozen, already-trained
            # energy model (loaded from ckpt_path). Freezing avoids the fresh-Adam
            # overshoot that NaN's a full fine-tune, and keeps the codebook fixed.
            # The discriminator is left trainable (its manual_backward must run); use
            # disc_weight=0 so it never touches the generator.
            for module in (self.encoder, self.decoder, self.quantize,
                           self.quant_conv, self.post_quant_conv):
                for p in module.parameters():
                    p.requires_grad = False
            for p in self.decoder.hit_head.parameters():
                p.requires_grad = True
            self.quantize.ema = False          # do not move the (frozen) codebook

    # ------------------------------------------------------------------ EMA
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    # -------------------------------------------------------------- encode/decode
    def encode(self, x, cond):
        h = self.encoder(x, cond)
        h = self.quant_conv(h)
        return self.quantize(h)

    @torch.no_grad()
    def encode_codes(self, x, cond):
        return self.encode(x, cond)["min_encoding_indices"]

    def decode(self, quant, cond):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant, cond)

    @torch.no_grad()
    def decode_codes(self, codes, cond):
        assert codes.dim() != 1
        z = self.quantize.get_codebook_entry(codes)
        z = self.post_quant_conv(z)
        return self.decoder(z, cond)

    @torch.no_grad()
    def decode_codes_fullchain(self, batch, pred, post=True, renorm=True, force_pred=False):
        ret = self.decode_codes(pred['codes_pred'], batch['cond'])
        pred.update(ret)
        if not post:
            return pred
        return self.postprocess(batch, pred, renorm=renorm, force_pred=force_pred)

    # -------------------------------------------------------------- (pre/post)process
    def preprocess_cond(self, batch):
        # E_inc ~ (N,1) in MeV (the dataloader converts condition keV->MeV). Current
        # DarkSHINE data is a single energy point (4 GeV = 4000 MeV); the conditioning
        # is kept general (log10 energy, centred ~0): log10(4000)-3.6 ~ 0.
        batch['log_E_inc'] = torch.log10(batch['E_inc']) - 3.6
        batch['cond'] = batch['log_E_inc']
        return batch

    def preprocess(self, batch):
        if 'pixels_E_orig' in batch:
            return batch
        batch = self.preprocess_cond(batch)

        E = batch['pixels_E']                                   # (N, 11, 43, 43)
        batch['pixels_E_orig'] = E.clone()
        Einc = batch['E_inc']
        if Einc.dim() < 2:
            Einc = Einc.unsqueeze(-1)
        pixels_R = E / Einc[..., None, None]                   # (N, 11, 43, 43)

        mask = self.mask[None]                                 # (1, 11, 43, 43)
        # R = E_deposited / E_incident summed over REAL crystals only (padding=0 anyway).
        R_true = (pixels_R * mask).sum(axis=(-1, -2, -3), keepdims=True)  # (N,1,1,1)
        batch['R_true'] = R_true
        # U: normalised shower shape, sums to 1 over real cells.
        batch['pixels_U'] = torch.nan_to_num(pixels_R / R_true.clamp(min=1e-12), nan=0.0, posinf=0.0, neginf=0.0)
        batch['pixels_R'] = pixels_R
        batch['pixels_E'] = batch['pixels_E_orig']
        return batch

    def _apply_hit_gate(self, U, hit_logits):
        # Hard hit gate: keep only cells with sigmoid(hit_logits) > threshold, then
        # renormalise the survivors back to sum-1 over real cells (preserves R/E_tot).
        # Falls back to the ungated U for any shower the gate would empty entirely.
        mask = self.mask[None].to(U.dtype)                      # (1, 11, 43, 43)
        gate = (torch.sigmoid(hit_logits) > self.hit_gate_threshold).to(U.dtype) * mask
        Ug = U * gate
        s = Ug.sum(dim=(-1, -2, -3), keepdim=True)
        return torch.where(s > 1e-8, Ug / s.clamp(min=1e-8), U)

    def forward(self, batch, test_code=False):
        enc = self.encode(batch['pixels_R'], batch['cond'])
        if test_code:
            pred = self.decode_codes(enc["min_encoding_indices"], batch['cond'])
        else:
            pred = self.decode(enc["quant"], batch['cond'])
        pred['qloss'] = enc["qloss"]
        pred['indices'] = enc["min_encoding_indices"]

        R = pred.get('R_pred', batch['R_true'])                # (N,1,1,1)
        Einc = batch['E_inc']
        if Einc.dim() < 2:
            Einc = Einc.unsqueeze(-1)
        # pixels_U_pred stays the ungated softmax (the reconstruction/disc loss compare
        # in U space). The hit gate is applied only to the energy outputs and only at
        # eval/generation, so training of the energy head is unchanged.
        U_energy = pred['pixels_U_pred']
        if self.hit_gate and (not self.training) and 'pixels_hit_logits' in pred:
            U_energy = self._apply_hit_gate(U_energy, pred['pixels_hit_logits'])
        pred['pixels_R_pred'] = U_energy * R
        pred['pixels_E_pred'] = pred['pixels_R_pred'] * Einc[..., None, None]
        return pred

    @torch.no_grad()
    def postprocess(self, batch, pred, renorm=True, force_pred=False):
        R = pred['R_pred'] if 'R_pred' in pred else batch['R_true']
        Einc = batch['E_inc']
        if Einc.dim() < 2:
            Einc = Einc.unsqueeze(-1)
        U = pred['pixels_U_pred']
        if self.hit_gate and 'pixels_hit_logits' in pred:
            U = self._apply_hit_gate(U, pred['pixels_hit_logits'])
        post = {'pixels_U_pred': U}
        post['pixels_R_pred'] = U * R
        post['pixels_E_pred'] = post['pixels_R_pred'] * Einc[..., None, None]
        if renorm:
            post = self.renorm_R(post)
        if self.convert_to_detector_shape:
            # map the (N,11,43,43) half-cell grid back to real detector crystals
            # (N,11,21,21). This is the ONNX / export output shape.
            for k in ('pixels_U_pred', 'pixels_R_pred', 'pixels_E_pred'):
                if k in post:
                    post[k] = downsample_to_crystals(post[k], self.downsample_mode)
        return post

    @torch.no_grad()
    def renorm_R(self, post):
        # U already sums to 1 over real cells (masked softmax), so this is ~identity;
        # kept for numerical safety / interface parity.
        factor = post['pixels_U_pred'].detach().sum(axis=(-1, -2, -3), keepdims=True).clip(min=1e-12)
        for k in ('pixels_U_pred', 'pixels_R_pred', 'pixels_E_pred'):
            if k in post:
                post[k] = post[k].detach() / factor
        return post

    # ----------------------------------------------------- physics metrics
    def _metric_epoch(self):
        # whether to emit histogram/figure metrics this validation epoch
        return self.do_metric and (self.current_epoch % self.metric_freq == 0)

    def on_validation_epoch_start(self):
        if self._metric_epoch():
            self._val_metrics = ShowerMetrics(self.downsample_mode, self.metric_hit_threshold)
        else:
            self._val_metrics = None

    @torch.no_grad()
    def _accumulate_metrics(self, b, pred):
        # stage-1 reco: pred is the reconstruction of the same shower (paired).
        if self._val_metrics is not None:
            self._val_metrics.update(b['pixels_E_orig'], pred['pixels_E_pred'], b['E_inc'])

    def on_validation_epoch_end(self):
        if self._val_metrics is not None and self._val_metrics.n > 0:
            self._val_metrics.compute_and_log(self, tag='reco',
                                              epoch=self.current_epoch, paired=True)
        self._val_metrics = None

    # -------------------------------------------------------------- training
    def get_last_layer(self):
        return self.decoder.get_adaptive_layer_weights()

    def training_step(self, batch, batch_idx):
        b = self.preprocess(batch)
        opt_ae, opt_disc = self.optimizers()
        pred = self(b)

        # optimizer 0: autoencoder (encoder/decoder/quantizer + adversarial g_loss)
        self.toggle_optimizer(opt_ae)
        aeloss, log_ae = self.loss(b, pred, 0, self.global_step,
                                   last_layer=self.get_last_layer(), split="train")
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()
        self.untoggle_optimizer(opt_ae)
        self.log_dict(log_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        # optimizer 1: discriminator
        self.toggle_optimizer(opt_disc)
        discloss, log_disc = self.loss(b, pred, 1, self.global_step,
                                       last_layer=self.get_last_layer(), split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)
        self.log_dict(log_disc, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers]
            for sch in schedulers:
                sch.step()

    def validation_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        b = self.preprocess(batch)
        pred = self(b, test_code=True)
        aeloss, log_ae = self.loss(b, pred, 0, self.global_step,
                                   last_layer=self.get_last_layer(), split="val" + suffix)
        discloss, log_disc = self.loss(b, pred, 1, self.global_step,
                                       last_layer=self.get_last_layer(), split="val" + suffix)
        rec_loss = log_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        del log_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self._accumulate_metrics(b, pred)
        return rec_loss

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.freeze_except_hit_head:
            ae_params = list(self.decoder.hit_head.parameters())
        else:
            ae_params = (list(self.encoder.parameters()) +
                         list(self.decoder.parameters()) +
                         list(self.quantize.parameters()) +
                         list(self.quant_conv.parameters()) +
                         list(self.post_quant_conv.parameters()))
        opt_ae = torch.optim.Adam(ae_params, lr=self.lr_g_factor * lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            sched = [
                {'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1},
                {'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1},
            ]
            return [opt_ae, opt_disc], sched
        return [opt_ae, opt_disc], []

    # image logging is geometry-specific; disabled for the xyz model (future work).
    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        return {}
