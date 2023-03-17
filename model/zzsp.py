from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from model.common import *
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np



class ZZSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.dropout = nn.Dropout(config.dropout)
        self.token_ids_attr, self.token_ids_obj, self.token_ids, self.soft_att, self.soft_obj, self.soft_att_obj, self.soft_prompt = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        self.loss_fn = CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.dtype = torch.float16
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        for p in self.parameters():
            p.requires_grad=False

        self.soft_att = nn.Parameter(self.soft_att)
        self.soft_obj = nn.Parameter(self.soft_obj)
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)

        # self.mlp = MLP(768, 768, 2, True, True, True, True, [1280, 512])

        self.weight = config.res_w


    def construct_soft_prompt(self):
        token_ids_obj = clip.tokenize("a photo of x",
                              context_length=self.config.context_length).cuda()
        token_ids_attr = clip.tokenize("a photo of x object",
                              context_length=self.config.context_length).cuda()
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)


        #### Construct the prompt for attributes
        tokenized_attr = torch.cat([clip.tokenize(tok, context_length=self.config.context_length) for tok in self.attributes])
        orig_token_embedding_attr = self.clip.token_embedding(tokenized_attr.cuda())
        soft_att = torch.zeros(
            (len(self.attributes), orig_token_embedding_attr.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding_attr):
            eos_idx = tokenized_attr[idx].argmax()
            soft_att[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        #### Construct the prompt for objects
        tokenized_obj = torch.cat([clip.tokenize(tok, context_length=self.config.context_length) for tok in self.classes])
        orig_token_embedding_obj = self.clip.token_embedding(tokenized_obj.cuda())
        soft_obj = torch.zeros(
            (len(self.classes), orig_token_embedding_obj.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding_obj):
            eos_idx = tokenized_obj[idx].argmax()
            soft_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)


        #### Construct the prompt for the prefix.
        prefix_init = "a photo of"
        n_ctx = len(prefix_init.split())
        prompt = clip.tokenize(prefix_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        prefix_vectors = embedding[0, 1 : 1 + n_ctx, :]
        return token_ids_attr, token_ids_obj, token_ids, soft_att, soft_obj, soft_att_obj, prefix_vectors



    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att = self.dropout(self.soft_att)
        soft_obj = self.dropout(self.soft_obj)
        # soft_att_obj = self.dropout(self.soft_att_obj)

        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_obj[
            obj_idx
        ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        normalized_x = x / x.norm(dim=-1, keepdim=True)
        return normalized_x


    def forward(self, batch, idx, status):
        batch_anchor, batch_attr, batch_obj, batch_target, pos_sample, neg_sample = batch
        batch_anchor, batch_attr, batch_obj, batch_target, pos_sample, neg_sample = batch_anchor.cuda(), batch_attr.cuda(), batch_obj.cuda(), batch_target.cuda(), pos_sample.cuda(), neg_sample.cuda()
        b = batch_anchor.shape[0]
        batch_img = torch.cat([batch_anchor, pos_sample, neg_sample], dim=0)



        #### Image Encoder
        batch_img = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768
        # batch_img_pos = self.visual(pos_sample.type(self.clip.dtype))
        # batch_img_neg = self.visual(neg_sample.type(self.clip.dtype))

        if status == "object":
            self.soft_obj.requires_grad = True
            self.soft_att.requires_grad = False
        elif status == "state":
            self.soft_obj.requires_grad = False
            self.soft_att.requires_grad = True
        else:
            self.soft_obj.requires_grad = True
            self.soft_att.requires_grad = True           
        
        # batch_img_mlp = self.mlp(batch_img.type(torch.float)).type(self.clip.dtype)
        # batch_img = self.weight * batch_img + (1 - self.weight) * batch_img_mlp

        #### Text Encoder
        token_tensors = self.construct_token_tensors(idx)
        text_features = self.text_encoder(self.token_ids, token_tensors, enable_pos_emb=self.enable_pos_emb)

        #### Compute Logits and loss
        logits = (self.clip.logit_scale.exp() * batch_img @ text_features.t())   
        # logits_pos = (self.clip.logit_scale.exp() * batch_img_pos @ text_features.t())  
        # logits_neg = (self.clip.logit_scale.exp() * batch_img_neg @ text_features.t())  
        
        logits = torch.split(logits, b)
        logits_anchor, logits_pos, logits_neg = logits[0], logits[1], logits[2]
        loss_contrastive = self.triplet_loss(logits_anchor, logits_pos, logits_neg)
        # print(loss_contrastive)        
        loss = self.loss_fn(logits_anchor, batch_target) + 1 * loss_contrastive

        return logits_anchor, loss
