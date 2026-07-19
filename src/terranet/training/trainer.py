"""LightningModule wiring FusionEncoder + ParamHead + optional DA branch."""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from terranet.models.da.coral import coral_loss
from terranet.models.da.dann import DomainDiscriminator, dann_lambda
from terranet.models.da.mmd import mmd2
from terranet.models.encoders.fusion import FusionEncoder
from terranet.models.hypernet import ParamHead

from .losses import total_loss


class TerraNetModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.encoder = FusionEncoder(
            desc_dim=cfg.model.desc_dim, emb_dim=cfg.model.emb_dim,
            use_descriptor=cfg.model.use_descriptor, use_raster=cfg.model.use_raster,
        )
        self.head = ParamHead(emb_dim=cfg.model.emb_dim)
        self.da_kind = cfg.model.get("da", "none")
        if self.da_kind == "dann":
            self.disc = DomainDiscriminator(cfg.model.emb_dim)
        self.register_buffer("target_scale", torch.tensor(cfg.model.target_scale))

    def forward(self, desc, raster):
        z = self.encoder(desc, raster)
        out = self.head(z)
        out["z"] = z
        return out

    def training_step(self, batch, batch_idx):
        (desc_s, ras_s, tgt_s, _), tb = batch["source"], batch.get("target")
        out = self(desc_s, ras_s)
        da_term = None
        if tb is not None and self.da_kind != "none":
            desc_t, ras_t, _, _ = tb
            zt = self.encoder(desc_t, ras_t)
            if self.da_kind == "coral":
                da_term = coral_loss(out["z"], zt)
            elif self.da_kind == "mmd":
                da_term = mmd2(out["z"], zt)
            elif self.da_kind == "dann":
                p = self.global_step / max(1, self.trainer.estimated_stepping_batches)
                lam = dann_lambda(p)
                logits = torch.cat([self.disc(out["z"], lam), self.disc(zt, lam)])
                labels = torch.cat([torch.zeros(len(out["z"])), torch.ones(len(zt))]).to(logits)
                da_term = F.binary_cross_entropy_with_logits(logits, labels)
        logs = total_loss(out, tgt_s, da=da_term, target_scale=self.target_scale,
                          w_da=self.hparams.model.get("w_da", 0.1))
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, prog_bar=True)
        return logs["loss"]

    def validation_step(self, batch, batch_idx):
        desc, ras, tgt, _ = batch
        out = self(desc, ras)
        mae = (out["mean"] - tgt).abs().mean(0)
        self.log_dict({"val/mae_gamma": mae[0], "val/mae_pl0": mae[1]}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.optim.lr,
                                weight_decay=self.hparams.optim.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.optim.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
