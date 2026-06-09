"""
NOTE: This is adapted from Andrej Karpathy's minGPT, via Xiulong's mod.

GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

import calo_ldm.layers.transformer as xfmr
from calo_ldm.util import instantiate_from_config, recursive_to
from calo_ldm.metrics import ShowerMetrics

from omegaconf import OmegaConf

from glob import glob
import os.path
import math

from functools import reduce

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, codebook_size, sequence_len, **kwargs):
        self.codebook_size = codebook_size
        self.sequence_len = sequence_len
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    """  the full GPT language model, with a context size of sequence_len """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.codebook_size, config.n_embd - config.cond_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.sequence_len, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # linear layer to transform skel latent size to match embedding size
        # self.linear = nn.Linear(512,)
        # transformer
        self.blocks = nn.Sequential(*[xfmr.Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.codebook_size, bias=False)

        self.sequence_len = config.sequence_len
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, cond, targets=None):
        # print("idx",idx.shape)
        # print("cond",cond.shape)
        b, t = idx.size()
        t += 1
        assert t <= self.sequence_len, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        # print("token_embeddings",token_embeddings.shape)
        # print("test",torch.cat([torch.zeros(b,1,token_embeddings.size(-1), device=idx.device),
        #                             token_embeddings], dim=1).shape)
        token_embedding_cat = torch.cat(
                                [torch.cat([torch.zeros(b,1,token_embeddings.size(-1), device=idx.device),
                                    token_embeddings], dim=1),
                                torch.cat([cond]*t, dim=1)],
                                dim=-1)

        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embedding_cat + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def sample(self, inp):
        b, t, _ = inp.size()
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # print(position_embeddings.shape, inp.shape)
        x = self.drop(inp + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class CondGPT(pl.LightningModule):
    def __init__(self, *,
            codebook_size,
            sequence_shape, # now shape(as list) is used instead of len, e.g. [32], 
            # sequence_len, # we use shaped codes for later usage. ds1: h, ds/3: h,w
            n_layer,
            n_head,
            vq_config,
            n_embd=512,
            #hidden_size=64,
            cond_bins=0, # number of bins for condition variable, if zero, don't bin.
            cond_dim=1, # guessing at what this means?
            cond_proj=False, # use Linear projection after embed for cond
            predict_R=False,
            R_seq_len=None, # How many bits or *codes to use for R if we are going to predict it
            R_bits=None, # How many *bits or codes to use for R if we are going to predict it
            R_renorm=True,  #renorm R after unpadding
            R_max=1.3,
            record_freq=1,           # run generation + emit gen-vs-truth metrics every N val epochs
            do_metric=True,          # log DarkSHINE generation-vs-truth physics metrics
            metric_hit_threshold=0.1,# [MeV] cell energy above which a cell is a "hit"
            downsample_mode='sum',   # 43x43 -> 21x21 crystal aggregation: 'sum' or 'min'
            convert_to_detector_shape=False,  # if True, generated showers are (N,11,21,21)
            monitor=None,
            use_vq_cond=True,
            ):
        super().__init__()
        # self.nonseg_dim=nonseg_dim
        self.codebook_size = codebook_size
        self.cond_bins = cond_bins
        self.predict_R = predict_R
        assert R_bits!=None or R_seq_len!=None
        if self.cond_bins == 0:
            cond_dim = 1
        self.use_vq_cond=use_vq_cond

        if monitor is not None:
            self.monitor = monitor

        if vq_config.get('logdir', None) is not None:
            vq_config['model_config'] = glob(os.path.join(vq_config['logdir'],'configs','*-project.yaml'))[-1]
            vq_config['checkpoint'] = glob(os.path.join(vq_config['logdir'],'checkpoints','epoch=*.ckpt'))[-1]
        self.vq_model = instantiate_from_config(OmegaConf.load(vq_config['model_config'])['model'],
                passthru={'ckpt_path': vq_config.get('checkpoint', None)}, overwrite=True)
        # freeze the VQ model
        for p in self.vq_model.parameters():
            p.requires_grad = False
        self.vq_model.eval()
        # detector-shape conversion is applied in vq_model.postprocess; push the
        # flags down so generation (-> sample_fullchain -> postprocess) honours them.
        self.downsample_mode = downsample_mode
        self.convert_to_detector_shape = convert_to_detector_shape
        self.vq_model.downsample_mode = downsample_mode
        self.vq_model.convert_to_detector_shape = convert_to_detector_shape
        assert self.codebook_size == self.vq_model.n_embed
        # assert sequence_len == self.vq_model.encoder.seq_out # better to check this...
        # assert sequence_len == self.vq_model.decoder.seq_in
        self.sequence_shape=sequence_shape
        assert len(sequence_shape)+1 in [2,3] # (batch + code grid): DarkSHINE uses a 2D (6,6) grid
        sequence_len = reduce(lambda x, y: x*y, sequence_shape)
        self.sequence_len=sequence_len
        self.input_dim = (11, 43, 43)  # DarkSHINE: depth, x, y

        if self.predict_R:
            self.bits_per_code = int(math.log2(self.codebook_size))
            assert 2**self.bits_per_code == self.codebook_size 
            # each code should embed one complete binary number
            if R_seq_len:
                self.R_seq_len=R_seq_len
                self.R_bits=self.R_seq_len*self.bits_per_code
            else:
                self.R_bits=R_bits
                self.R_seq_len = math.ceil(self.R_bits/self.bits_per_code)
            # 
            print(f"Using {self.R_seq_len} initial sequence elements to predict R of {self.R_bits} bits")
            sequence_len = sequence_len + self.R_seq_len

        self.config = GPTConfig(
                codebook_size=codebook_size, sequence_len=sequence_len,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                cond_dim=cond_dim,
                )
        self.gpt = GPT(self.config)
        self.cond_proj=cond_proj
        if self.cond_bins > 0:
            self.lab_emb = nn.Embedding(self.cond_bins, cond_dim) # cond_bins = 100, hid = 64
            self.proj = nn.Linear(cond_dim, cond_dim, bias=False) # hidden_size = cond_dim = 16
        else:
            self.register_module('lab_emb', None)
            self.register_module('proj', None)

        self.n_embd=n_embd
        self.R_renorm=R_renorm
        self.R_max=R_max
        self.record_freq=record_freq
        self.do_metric=do_metric
        self.metric_hit_threshold=metric_hit_threshold
        self._val_metrics=None

        # loopback test of R coding (test within the representable [0, R_max) range)
        s=torch.rand([1000,1000]) * self.R_max
        s_codes=self.convertR(s)
        s_rep=self.decodeR(s_codes)
        print("loopback test R: error sum",(s_rep-s).sum()/s.sum())
        if(abs((s_rep-s).sum()/s.sum())>0.01):
            print("? do you have a good R enbemding function?")
            assert False

    def forward(self, cond, idx, targets): #skel_input
        if self.cond_bins:
            cond = self.lab_emb(cond)
            if self.cond_proj:
                cond = self.proj(cond)
            logits, loss = self.gpt(idx, cond, targets)
        else:
            cond = cond.unsqueeze(-2) # --> (N,dumm,1)
            logits, loss = self.gpt(idx, cond, targets)
        return logits, loss
    
    def convertR(self,R_in): # convert R(float) to N-base number, N=n_embed
        if R_in.dim()!=2:
            raise NotImplementedError(f"Wrong R dim {R_in.shape}")
        # encode R as a fixed-point representation in the first log2(b) steps
        # of the latent sequence. the inverse of this process is in postprocess_codes
        R = ((2**self.R_bits)*R_in/self.R_max).long()
        R = torch.clip(R, 0, 2**self.R_bits-1)
        mask = 2**self.bits_per_code - 1
        rcodes = []
        R = R.unsqueeze(-1)
        for _ in range(self.R_seq_len):
            rcodes.append(R & mask)
            R = R >> self.bits_per_code
        return torch.concat(rcodes[::-1], axis=-1).reshape(R_in.shape[0],-1) # reorder as msb to lsb
    
    def decodeR(self,_codes):
        if _codes.dim()!=2:
            raise NotImplementedError(f"Wrong codes dim {_codes.shape}")
        codes = _codes.reshape(_codes.shape[0],-1,self.R_seq_len)
        R_pred = codes[:,:,0]
        for i in range(self.R_seq_len-1):
            R_pred = R_pred << self.bits_per_code
            R_pred = R_pred | codes[:,:,i+1]
        return (R_pred.double() / 2**self.R_bits * self.R_max).float().reshape(_codes.shape[0],-1) # GPT_R mustbe 2D

    def preprocess_cond(self,batch):
        # deal with cond 
        if self.cond_bins == 0: # unbinned cond
            # take log and normalized to range [-1, 1]
            batch['gpt_cond'] = batch['log_E_inc']
        else:
            if not self.use_vq_cond: # allow gpt-cond is different as vq
                batch['gpt_cond'] = ((torch.log10(self.batch['E_inc'].squeeze(-1))-3)/3*self.cond_bins).long()
            else:
                # use directly VQ cond
                batch['gpt_cond']=batch['cond']
        return batch
    
    def preprocess(self, batch): 
        # difference between R_true and gpt_R_true ?
        # R_true is the same shape as E
        # gpt_R_true always (N,1)
        if 'gpt_R_true' in batch: 
            return batch # prevent double processing
        
        batch = self.vq_model.preprocess(batch)
        batch = self.preprocess_cond(batch)

        if not self.predict_R:
            return batch
        
        # deal with R_true
        batch['gpt_R_true'] = batch.pop('R_true').squeeze()
        if batch['gpt_R_true'].dim()==1: # back-compatible with no-layer-norm ds1
            batch['gpt_R_true']=batch['gpt_R_true'].unsqueeze(-1)
        elif batch['gpt_R_true'].dim()!=2:
            raise NotImplementedError(f"Wrong R_true dimension {batch['gpt_R_true'].shape}")
        # N,R,Z,A or N,X --> N,Z
        
        # convert R into codes (single global R for the xyz model)
        batch['R_codes'] = self.convertR(batch['gpt_R_true'])
        return batch

    def postprocess_codes(self, codes): 
        if isinstance(codes, list): # gpt is generate each code, need concat
            codes = torch.cat(codes, axis=1)

        ret = {}
        if self.predict_R:
            # single global R: the first R_seq_len codes encode R, the rest are the code grid.
            ret['gpt_R_unique_preds'] = self.decodeR(codes[:,:self.R_seq_len]) # GPT_R mustbe 2D
            codes = codes[:,self.R_seq_len:]
            ret['R_pred'] = ret['gpt_R_unique_preds'].reshape(
                -1 , # batch
                *((1,) * (len(self.sequence_shape) +1 )) # 2D latent grid + channel (1)
                ) # R_pred broadcasts over the decoded shower

        # recover shape
        ret['codes_pred'] = self.unflatten_codes(codes)

        return ret

    # the reason use two type of codes is to ensure the VQ works as expected!
    # prevent any wired shape bug
    def flatten_codes(self,vq_codes):
        ret=vq_codes.reshape(-1,self.sequence_len)
        # assert ret.shape[0] == batch_size
        return ret
    def unflatten_codes(self,GPT_codes): 
        ret=GPT_codes.reshape(-1,*self.sequence_shape)
        # assert ret.shape[0] == batch_size
        return ret
    
    def trainval_step(self, batch, batch_idx, split):
        self.vq_model.eval()
        with torch.no_grad():
            batch = self.preprocess(batch)
            codes = self.vq_model.encode_codes(batch['pixels_R'], batch['cond']) # codes in sane shape
            # codes = self.vq_model.predict_codes(batch,debug=False)['min_encoding_indices']
            # flatten to GPT codes (N,*)
            # print(codes.shape)
            codes = self.flatten_codes(codes)
            # codes = codes.reshape(batch['E_inc'].shape[0] ,-1) # flatten to (N,*)

            if self.predict_R:
                codes = torch.cat([batch['R_codes'], codes], axis=1) # (*, R_len + H*W)

        idx, targets = codes[:,:-1], codes
        logits, loss = self(batch['gpt_cond'], idx, targets)

        if split == 'train':
            self.log(f"train/loss", loss, on_step=True, on_epoch=True)
        else:
            self.log(f"{split}/loss", loss, on_step=False, on_epoch=True)

        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.trainval_step(batch, batch_idx, split='train')
        return loss
    
    # ----------------------------------------------------- physics metrics
    def _metric_epoch(self):
        # generation is the expensive part; only run it every record_freq epochs.
        return (self.do_metric
                and (not self.trainer.sanity_checking)
                and (self.current_epoch % self.record_freq == 0))

    def on_validation_epoch_start(self):
        if self._metric_epoch():
            self._val_metrics = ShowerMetrics(self.downsample_mode, self.metric_hit_threshold)
        else:
            self._val_metrics = None

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        logits, loss = self.trainval_step(batch, batch_idx, split='val')

        # generation metrics: sample showers from the GPT and compare to truth.
        if self._val_metrics is not None:
            batch = self.preprocess(batch)
            gen_post = self.sample_fullchain(batch)   # {'pixels_E_pred': (N,11,43,43), ...}
            self._val_metrics.update(batch['pixels_E_orig'], gen_post['pixels_E_pred'], batch['E_inc'])

        return loss

    def on_validation_epoch_end(self):
        if self._val_metrics is not None and self._val_metrics.n > 0:
            # generation is unpaired (independent samples) -> distribution comparison.
            self._val_metrics.compute_and_log(self, tag='gen',
                                              epoch=self.current_epoch, paired=False)
        self._val_metrics = None

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        return opt
    
    def sample(self, cond, steps=None, temperature=1.0, sample=True, top_k=None): #
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of sequence_len, unlike an RNN that has an infinite context window.
        """
        sequence_len = self.gpt.sequence_len
        if steps is None:
            steps = sequence_len

        with torch.no_grad():
            # input cond = (N,1)
            if self.cond_bins:
                cond = self.lab_emb(cond) # auto add one more dim
                # honestly I don't know why we had a projection at all before?
                if self.cond_proj:
                    # print(self.proj.weight.shape)
                    cond = self.proj(cond)
                    #print('cond_proj', cond.shape)
            else:
                cond = cond.unsqueeze(-2)
            # cond = (N,1,1)
            self.gpt.eval()
            # x = skel_latent
            # print("BAD shape1",cond.shape)
            x = torch.cat([torch.zeros((cond.shape[0], 1, self.n_embd-cond.shape[-1]), device=cond.device), cond], dim=2) # concat every step
            indices = []
            for k in range(steps):
                x_cond = x if x.size(1) <= sequence_len else x[:, -sequence_len:]  # crop context if needed
                # print("cond shape",x_cond.shape)
                logits = self.gpt.sample(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                indices.append(ix)
                emb_ix = self.gpt.tok_emb(ix)
                # emb_ix += skel_latent # add at every steps
                emb_ix = torch.cat([emb_ix, cond], dim=-1) # concat at every steps
                # append to the sequence and continue
                x = torch.cat((x, emb_ix), dim=1)

        return self.postprocess_codes(indices)

    @torch.no_grad()
    def sample_fullchain(self,batch):
        gen = self.sample(batch['gpt_cond']) 
        gen_post = self.vq_model.decode_codes_fullchain(batch, gen, post=True, renorm=self.R_renorm, force_pred=True) # ensure R_pred is used instead of R_true
        return gen_post

    @staticmethod
    def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out
