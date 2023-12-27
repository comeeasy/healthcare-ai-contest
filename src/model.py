import lightning.pytorch as pl

from torch import nn
from torchmetrics.classification import Accuracy, F1Score

from cfg import CFG

import torch
import torch.nn as nn
import torch.optim as optim
import timm


class CNNModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = self.build_model()
        self.fc = nn.Linear(768 + 5 + 32, 2, bias=True) # tooth_pos, tooth_num one-hot vec
        
        self.label_smoothing = CFG.label_smoothing
        
        # Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # hyper parameters
        self.hparams['label_smoothing'] = self.label_smoothing
        
        # hyperparameters
        self.hparams["lr"] = CFG.lr
        self.hparams["optim_betas"] = CFG.optim_betas
        self.hparams["optim_eps"] = CFG.optim_eps
        self.hparams["optim_weight_decay"] = CFG.optim_weight_decay
        
        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.train_f1 = F1Score(task="multiclass", num_classes=2)
        self.valid_acc = Accuracy(task="multiclass", num_classes=2)
        self.valid_f1 = F1Score(task="multiclass", num_classes=2)
        self.valid_f1_weighted = F1Score(task="multiclass", num_classes=2, average='weighted')
        self.test_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_f1 = F1Score(task="multiclass", num_classes=2)
        self.test_f1_weighted = F1Score(task="multiclass", num_classes=2, average='weighted')
    
        # save hyperparameters
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
            lr=self.hparams["lr"], betas=self.hparams["optim_betas"], 
            eps=self.hparams["optim_eps"], weight_decay=self.hparams["optim_weight_decay"])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-8)
        
        return [optimizer], [lr_scheduler]
        
    
    def build_model(self):
        checkpoint_path = "/zz1236zz/.cache/huggingface/hub/models--timm--tf_efficientnet_b3.ns_jft_in1k/snapshots/fd765843d68fcbba2757c8cf89e810efc350bc7d/model.safetensors"
        model = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', pretrained=True, checkpoint_path=checkpoint_path)
        model.classifier = nn.Linear(in_features=1536, out_features=768, bias=True)
        
        return model
    
    def forward(self, imgs, tooth_num_one_hot, tooth_pos_one_hot):
        imgs = self.batchfy(imgs)
        tooth_num_one_hot = self.batchfy(tooth_num_one_hot)
        tooth_pos_one_hot = self.batchfy(tooth_pos_one_hot)
        
        outputs = self.model(imgs)
        
        logits = torch.concat([outputs, tooth_num_one_hot, tooth_pos_one_hot], dim=1)
        
        y_hat = self.fc(logits)
        
        return y_hat
    
    def predict(self, imgs, tooth_num_one_hot, tooth_pos_one_hot):
        imgs = imgs.squeeze(0)
        tooth_num_one_hot = tooth_num_one_hot.squeeze(0)
        tooth_pos_one_hot = tooth_pos_one_hot.squeeze(0)
        
        outputs = self.model(imgs)
        
        logits = torch.concat([outputs, tooth_num_one_hot, tooth_pos_one_hot], dim=1)
        
        y_hat = self.fc(logits)
        
        return y_hat
    
    def batchfy(self, inputs): # [B x 2 x ...] -> [2B x ...]
        input_list = []
        batch_size = inputs.shape[0] # B might be different when validation step
        for b in range(batch_size):
            for i in range(2): # sample 2 tooth infos from dataset
                input_list.append(inputs[b, i])
        return torch.stack(input_list)
    
    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y = self.batchfy(Y)
        Y = Y.squeeze(1)
    
        logits = self(*X) # [B x 2]
        loss = self.cross_entropy_loss(logits, Y)
        
        self.train_acc(logits, Y)
        self.train_f1(logits, Y)
        self.log("train/acc", self.train_acc, on_step=True, sync_dist=True)
        self.log("train/f1", self.train_f1, on_step=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y = self.batchfy(Y) # [B x 1]
        Y = Y.squeeze(1)

        logits = self(*X)
        
        loss = self.cross_entropy_loss(logits, Y)
        
        self.valid_acc(logits, Y)
        self.valid_f1(logits, Y)
        self.valid_f1_weighted(logits, Y)
        self.log("val/accuracy", self.valid_acc, on_step=True, sync_dist=True)
        self.log("val/f1", self.valid_f1, on_step=True, sync_dist=True)
        self.log("val/f1_weighted", self.valid_f1_weighted, on_step=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.squeeze(0, 2)
    
        logits = self.predict(*X)
        
        self.test_acc(logits, Y)
        self.test_f1(logits, Y)
        self.test_f1_weighted(logits, Y)
        self.log("test/accuracy", self.test_acc, on_epoch=True, sync_dist=True)
        self.log("test/f1", self.test_f1, on_epoch=True, sync_dist=True)
        self.log("test/f1_weighted", self.test_f1_weighted, on_epoch=True, sync_dist=True)
    
    def on_train_epoch_end(self) -> None:
        self.train_acc.reset()
        self.valid_acc.reset()
        
    def initialize_linear_layer(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)