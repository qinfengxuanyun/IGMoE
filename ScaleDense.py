import torch
import torch.nn as nn
import math
from transformer import  TransformerEncoder
from ScaleDense import ScaleDense_VAE, GeneResEncoder,GeneMLPEncoder,GeneDesEncoder
from torch.nn import functional as F
from pos_embed import get_3d_sincos_pos_embed
from mamba_model import BiMamba_block

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(8, 8, 8), emb_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = F.pad(x,(3,2,2,2,0,0),'constant',torch.min(x))
        x = self.projection(x)  # (B, emb_size, D', H', W')
        # x = x.flatten(2)        # (B, emb_size, N_patches)
        # x = x.transpose(1, 2)   # (B, N_patches, emb_size)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 如果输入和输出通道不匹配，需要在捷径路径中添加一个卷积层进行匹配
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels),
                nn.MaxPool3d(2,2)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool3d(out,2) 
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet3DEncoder(nn.Module): 
    def __init__(self, in_channels=1, base_channels=32):
        super(ResNet3DEncoder, self).__init__()
        # 残差模块
        self.layer1 = ResidualBlock3D(1, base_channels, stride=2)  # 下采样2倍
        self.layer2 = ResidualBlock3D(base_channels, base_channels * 4, stride=2)  # 再次下采样2倍
        self.layer3 = ResidualBlock3D(base_channels * 4, base_channels * 16, stride=2)  # 最后下采样2倍

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
  
class PatchEmbedding3DResnet(nn.Module):
    def __init__(self, in_channels=1, patch_size=(8, 8, 8), emb_size=512):
        super().__init__()
        self.patch_size = patch_size
        # self.projection = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.encoder = ResNet3DEncoder()#ScaleDense_VAE()

    def patchify(self,imgs,p=8):
        """
        imgs: (N, 1, H, W,L)
        x: (N, L, patch_size**3 *1)
        """
        imgs = F.pad(imgs,(3,2,2,2,0,0),'constant',torch.min(imgs))
        w,h,l = imgs.shape[2] // p,imgs.shape[3] // p,imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, w, p, h, p, l, p))
        x = torch.einsum('ncwohplq->nwhlopqc', x)
        x = x.reshape(shape=(imgs.shape[0], w,h,l, p**3 *1))
        return x
    
    def forward(self, x):
        # feature = self.patch_embed(x)
        feature = self.patchify(x)
        B,W,H,L,C = feature.shape
        feature = feature.view(B,-1,C)
        feature = self.encoder(feature.view(B*feature.shape[1],1,8,8,8)).view(B,W,H,L,-1).permute([0,4,1,2,3])
        return feature

class dense_layer(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
    def forward(self,x):
        new_features = self.block(x)
        x = F.max_pool3d(x,2) 
        x = torch.cat([new_features,x], 1)
        return x
    
class DenseNet3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(DenseNet3DEncoder, self).__init__()
        self.block, last_channels = self._make_block(base_channels,3)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels,512,(1,1,1),stride=1,padding=0,bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            )
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1
        for i in range(nb_block):         
            outchannels = nb_filter * pow(2,i+1) 
            blocks.append(dense_layer(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    def forward(self, x):
        out = self.block(x)
        out =  self.out(out)
        return out
  
class PatchEmbedding3DDensenet(nn.Module):
    def __init__(self, in_channels=1, patch_size=(8, 8, 8), emb_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = DenseNet3DEncoder()

    def patchify(self,imgs,p=8):
        """
        imgs: (N, 1, H, W,L)
        x: (N, L, patch_size**3 *1)
        """
        imgs = F.pad(imgs,(3,2,2,2,0,0),'constant',torch.min(imgs))
        w,h,l = imgs.shape[2] // p,imgs.shape[3] // p,imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, w, p, h, p, l, p))
        x = torch.einsum('ncwohplq->nwhlopqc', x)
        x = x.reshape(shape=(imgs.shape[0], w,h,l, p**3 *1))
        return x
    
    def forward(self, x):
        # feature = self.patch_embed(x)
        feature = self.patchify(x)
        B,W,H,L,C = feature.shape
        feature = feature.view(B,-1,C)
        feature = self.encoder(feature.view(B*feature.shape[1],1,8,8,8)).view(B,W,H,L,-1).permute([0,4,1,2,3])
        return feature

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings=1201, hidden_size=512):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_embeddings2 = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        #self.dropout_embedding = nn.Dropout2d(0.2)

    def forward(self,inputs_embeds=None,past_key_values_length=0 ):
        seq_length = inputs_embeds.shape[1]
        position_ids = self.position_ids[:, past_key_values_length:past_key_values_length+seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = inputs_embeds
        embeddings = position_embeddings * embeddings + position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class Embeddings3D(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings=1201, hidden_size=512):
        super().__init__()
        self.position_embeddings  = torch.from_numpy(get_3d_sincos_pos_embed(hidden_size, (11,13,11), cls_token=True)).float().unsqueeze(0).cuda()
        self.position_embeddings.requires_grad=False
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        #self.dropout_embedding = nn.Dropout2d(0.2)

    def forward(self,inputs_embeds=None,past_key_values_length=0 ):
        seq_length = inputs_embeds.shape[1]
        position_ids = self.position_ids[:, past_key_values_length:past_key_values_length+seq_length][0,:].cpu().numpy().tolist()
        position_embeddings = self.position_embeddings[:,position_ids,:]
        embeddings = inputs_embeds
        embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class VisionTransformerMAE(nn.Module): 
    def __init__(self, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = 11*13*11
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=False,M_Ratio=0.75):
        # Patch embedding
        x = F.pad(x,(1,0,2,1,2,2),'replicate')
        feature = self.patch_embed(x)
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])
        
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, :, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle

        feature_ = self.TE(feature_)
        feature_ = torch.tanh(feature_)

        if out_rec: 
            x_list = []
            x_ = self.decoder_pred(feature_)
            x_ = self.unpatchify(x_,(B,C,W,H,L))[:,:,2:-2,2:-1,1:]
            x_list.append(x_)
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask, x_list
        
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask

class VisionTransformerMAE2(nn.Module): 
    def __init__(self, in_channels=1, img_size=(80, 100, 83), patch_size=(8, 8, 8), emb_size=1024, dropout=0.1):
        super().__init__() 
        self.patch_size = patch_size
        self.patch_num = 10*13*11#11*13*11
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.TE2 = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=False,M_Ratio=0):
        # Patch embedding
        feature = self.patch_embed(x)
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])
        
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append cls token
        cls_token = self.pos_embed(self.cls_token,0)
        cls_tokens = cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)
        feature_m = self.TE(feature_m)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, 1:, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle
        feature_ = torch.cat([feature_m[:, :1, :], feature_], dim=1)  # append cls token
        # add pos embed
        feature_ = self.pos_embed2(feature_,0)
        feature_ = self.TE2(feature_)
        feature_ = feature_[:, 1:, :]
        if out_rec: 
            x_list = []
            x_ = self.decoder_pred(feature_)
            x_ = self.unpatchify(x_,(B,C,W,H,L))[:,:,:,2:-2,3:-2]
            x_list.append(x_)
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask, x_list
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask,feature_

class VisionTransformerMAE3(nn.Module): 
    def __init__(self, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512, dropout=0.1,nb_filter=32):
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = 10*12*10
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Upsample((21,25,21)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Upsample((42,50,43)),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((84,101,87) ),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=False,M_Ratio=0.75):
        # Patch embedding
        # x = F.pad(x,(1,0,2,1,2,2),'replicate')
        feature = self.patch_embed(x)
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])
        
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, :, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle

        # add pos embed
        feature_ = self.TE(feature_)
        feature_ = torch.tanh(feature_)

        if out_rec: 
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            x_1 =  self.up1(feature_)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            #x_  = self.out4(x_3)
            #return feature, x_
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature, mask, x_list
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask

class VisionTransformerMAE4(nn.Module): 
    def __init__(self, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512, dropout=0.1,nb_filter=32):
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = 10*12*10
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.TE = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.TE2 = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

        # (80,100,83)(40,50,41) (20,25,20) (10,12,10)

        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Upsample((20,25,20)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Upsample((40,50,41)),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((80,100,83)),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
        )
        
        # self.up1 = nn.Sequential(
        #     nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
        #     nn.BatchNorm3d(nb_filter*4),
        #     nn.ELU(),     
        #     nn.Upsample((21,25,21)),
        #     nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter*4),
        #     nn.ELU(),
        # )
        # self.up2 = nn.Sequential(
        #     nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
        #     nn.BatchNorm3d(nb_filter*2),
        #     nn.ELU(),
        #     nn.Upsample((42,50,43)),
        #     nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter*2),
        #     nn.ELU(),
        # )
        # self.up3 = nn.Sequential(
        #     nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.BatchNorm3d(nb_filter),
        #     nn.ELU(),
        #     nn.Upsample((84,101,87) ),
        #     nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter),
        #     nn.ELU(),
        # )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=False,M_Ratio=0.75):
        # Patch embedding
        # x = F.pad(x,(1,0,2,1,2,2),'replicate')
        feature = self.patch_embed(x)
        feature = torch.tanh(feature)

        B,C,W,H,L = feature.shape
        feature = feature.view(B,C,-1).permute([0,2,1])
        
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
        # append cls token
        cls_token = self.pos_embed(self.cls_token,0)
        cls_tokens = cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)
        feature_m = self.TE(feature_m)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, 1:, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle
        feature_ = torch.cat([feature_m[:, :1, :], feature_], dim=1)  # append cls token
        # add pos embed
        feature_ = self.pos_embed2(feature_,0)
        feature_ = self.TE2(feature_)
        feature_ = feature_[:, 1:, :]
        feature_ = torch.tanh(feature_)
        if out_rec: 
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            x_1 =  self.up1(feature_)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            #x_  = self.out4(x_3)
            #return feature, x_
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature, mask, x_list
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature, mask

class VisionTransformerMAE5(nn.Module): 
    def __init__(self, in_channels=1, img_size=(84, 101, 87), patch_size=(8, 8, 8), emb_size=512, dropout=0.1,nb_filter=32):
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = 11*13*11#10*13*11#
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.TE = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.TE2 = TransformerEncoder(seq_len=self.patch_num+1, d_model=emb_size, num_heads=8, 
                            num_layers=3, dim_feedforward=1024)
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

        self.encoder = ScaleDense_VAE()

        # self.up1 = nn.Sequential(
        #     nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
        #     nn.BatchNorm3d(nb_filter*4),
        #     nn.ELU(),     
        #     nn.Upsample((20,25,20)),
        #     nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter*4),
        #     nn.ELU(),
        # )
        # self.up2 = nn.Sequential(
        #     nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
        #     nn.BatchNorm3d(nb_filter*2),
        #     nn.ELU(),
        #     nn.Upsample((40,50,41)),
        #     nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter*2),
        #     nn.ELU(),
        # )
        # self.up3 = nn.Sequential(
        #     nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
        #     #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.BatchNorm3d(nb_filter),
        #     nn.ELU(),
        #     nn.Upsample((80,100,83)),
        #     nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
        #     nn.BatchNorm3d(nb_filter),
        #     nn.ELU(),
        # )
        
        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Upsample((21,25,21)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Upsample((42,50,43)),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((84,101,87) ),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def patchify(self,imgs,p=8):
        """
        imgs: (N, 1, H, W,L)
        x: (N, L, patch_size**3 *1)
        """
        w,h,l = imgs.shape[2] // p,imgs.shape[3] // p,imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, w, p, h, p, l, p))
        x = torch.einsum('ncwohplq->nwhlopqc', x)
        x = x.reshape(shape=(imgs.shape[0], w,h,l, p**3 *1))
        return x
    
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        _,_,w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        return imgs
    
    def forward(self, x, out_rec=False,M_Ratio=0.75):
        # Patch embedding
        # x = F.pad(x,(3,2,2,2,0,0),'replicate')
        x = F.pad(x,(1,0,2,1,2,2),'replicate')
        # feature = self.patch_embed(x)
        feature = self.patchify(x)
        B,W,H,L,C = feature.shape
        feature = feature.view(B,-1,C)
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,0), M_Ratio)
        feature_m = self.encoder(feature_m.view(B*feature_m.shape[1],1,8,8,8)).view(B,feature_m.shape[1],-1)
        feature_m = torch.tanh(feature_m)
        # append cls token
        cls_token = self.pos_embed(self.cls_token,0)
        cls_tokens = cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)
        feature_m = self.TE(feature_m)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(feature_m.shape[0], ids_restore.shape[1] + 1 - feature_m.shape[1], 1)
        feature_ = torch.cat([feature_m[:, 1:, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle
        feature_ = torch.cat([feature_m[:, :1, :], feature_], dim=1)  # append cls token
        
        # add pos embed
        feature_ = self.pos_embed2(feature_,0)
        feature_ = self.TE2(feature_)
        feature_ = feature_[:, 1:, :]
        feature_ = torch.tanh(feature_)
        
        if out_rec: 
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            x_1 =  self.up1(feature_)
            x_2 =  self.up2(x_1)
            x_3 =  self.up3(x_2)
            #x_  = self.out4(x_3)
            #return feature, x_
            x_list = []
            x_list.append(self.out1(x_1))
            x_list.append(self.out2(x_2))
            x_list.append(self.out3(x_3))  
            return feature_, mask, x_list
        else:
            feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
            feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
            return feature_, mask
        
        # if out_rec: 
        #     x_list = []
        #     x_ = self.decoder_pred(feature_)
        #     x_ = self.unpatchify(x_,(B,C,W,H,L))[:,:,2:-2,2:-1,1:]
        #     x_list.append(x_)
        #     feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
        #     feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
        #     return feature_, mask, x_list
        # else:
        #     feature_  = feature_.permute([0,2,1]).view(B,C,W,H,L)
        #     feature  = feature.permute([0,2,1]).view(B,C,W,H,L)
        #     return feature_, mask

class MRIMambaMAE(nn.Module): 
    def __init__(self, config, in_channels=1, img_size=(80,100,83), patch_size=(8, 8, 8), emb_size=512, dropout=0.1,nb_filter=32):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_num = config.max_position_embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.encoder = DenseNet3DEncoder()#ResNet3DEncoder()#

        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*patch_size[1]*patch_size[2]*in_channels, bias=True)
        
        self.up1 = nn.Sequential(
            nn.Conv3d(512,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),      
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),     
            nn.Upsample((20,25,20)),
            nn.Conv3d(nb_filter*4,nb_filter*4,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter*4,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
            nn.Upsample((40,50,41)),
            nn.Conv3d(nb_filter*2,nb_filter*2,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter*2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter*2,nb_filter,(3,3,3),stride=1,padding=1,bias=False), 
            #nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((80,100,83)),
            nn.Conv3d(nb_filter,nb_filter,(3,3,3),stride=1,padding=1,bias=False),     
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
        )
        
        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*4,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter*2,1,(3,3,3),stride=1,padding=0,bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3/2)), 
                                  nn.Conv3d(nb_filter,1,(3,3,3),stride=1,padding=0,bias=False))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.mamba =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.mamba2 =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)
        self.pos_embed2 = Embeddings(self.patch_num+1,emb_size)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking2(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(1, L, device=x.device).repeat(N,1)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0-mask.unsqueeze(2))

        return x_masked, mask
   
    def patchify(self,imgs,p=8):
        """
        imgs: (N, 1, H, W,L)
        x: (N, L, patch_size**3 *1)
        """
        imgs = F.pad(imgs,(3,2,2,2,0,0),'replicate')
        w,h,l = imgs.shape[2] // p,imgs.shape[3] // p,imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, w, p, h, p, l, p))
        x = torch.einsum('ncwohplq->nwhlopqc', x)
        x = x.reshape(shape=(imgs.shape[0], w,h,l, p**3 *1))
        return x
    
    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p =  self.patch_size[0]
        w,h,l =feature_shape
        
        x = x.reshape(shape=(x.shape[0], w,h,l, p, p, p,1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        imgs = imgs[:,:,:,2:-2,3:-2]
        return imgs
    
    def forward(self, x=None, out_rec=False,M_Ratio=0.75):
        if out_rec:
            latent, mask,ids_restore = self.forward_encoder(x, M_Ratio)
            x_ = self. forward_decoder(latent,ids_restore)
            # x_ = self.unpatchify(x_,(B,C,L))[:,:,0:-pad_len]
            return latent, mask, x_
        else: 
            feature = self.forward_(x)
            return feature

    def forward_encoder(self, x=None, M_Ratio=0.75):
        # Patch embedding
        #(80,100,83) (40,50,41)  (20,25,20) (10,12,10)
        feature = self.patchify(x)
        B,W,H,L,C = feature.shape
        feature = feature.view(B,-1,C)
        feature = self.encoder(feature.view(B*feature.shape[1],1,self.patch_size[0],self.patch_size[1],self.patch_size[2])).view(B,feature.shape[1],-1)
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature,1), M_Ratio)
       # append cls token
        cls_tokens = self.cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)
        
        for encoder in self.mamba:
            feature_m = encoder(feature_m)
        latent  = feature_m#.permute([0,2,1])

        return latent, mask, ids_restore
    
    def forward_decoder(self, latent=None,ids_restore=None):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(latent.shape[0], ids_restore.shape[1] + 1 - latent.shape[1], 1)
        feature_ = torch.cat([latent[:, 1:, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle
        feature_ = torch.cat([latent[:, :1, :], feature_], dim=1)  # append cls token
        
        # add pos embed
        feature_ = self.pos_embed2(feature_,0)
        for encoder in self.mamba2:
            feature_ = encoder(feature_)
        feature_ = feature_[:, 1:, :]
        w,h,l = math.ceil(1.0*self.img_size[0]/ self.patch_size[0]),math.ceil(1.0*self.img_size[1]/ self.patch_size[1]),math.ceil(1.0*self.img_size[2]/ self.patch_size[2])
        # feature_ = feature_.permute([0,2,1]).reshape(feature_.shape[0],feature_.shape[2],w,h,l)
        # x_1 =  self.up1(feature_)
        # x_2 =  self.up2(x_1)
        # x_3 =  self.up3(x_2)
        # x_ = self.out3(x_3)
        x_ = self.decoder_pred(feature_)
        x_ = self.unpatchify(x_, (w,h,l))
        return x_

    def forward_(self, x=None, M_Ratio=0):
        # Patch embedding
        #(80,100,83) (40,50,41)  (20,25,20) (10,12,10)
        feature = self.patchify(x)
        B,W,H,L,C = feature.shape
        feature = feature.view(B,-1,C)
        feature = self.encoder(feature.view(B*feature.shape[1],1,self.patch_size[0],self.patch_size[1],self.patch_size[2])).view(B,feature.shape[1],-1)
        if M_Ratio!= 0:
            feature, mask = self.random_masking2(feature,M_Ratio)
        return feature
 
class GeneMambaMAE(nn.Module): 
    def __init__(self, config, in_channels=3, patch_size=(16,), emb_size=512, dropout=0.1,nb_filter=32):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.patch_num = config.max_position_embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.encoder = GeneMLPEncoder()#GeneMLPEncoder()#
        self.decoder_pred = nn.Linear(emb_size, patch_size[0]*in_channels, bias=True)
        self.mamba =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.pos_embed = Embeddings(self.patch_num+1,emb_size)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, D, L], sequence
        """
        N, D, L = x.shape  # batch, dim, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0-mask.unsqueeze(1))

        return x_masked, mask
    
    def random_masking2(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, D, L], sequence
        """
        N, D, L = x.shape  # batch, dim, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(1, L, device=x.device).repeat(N,1)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0-mask.unsqueeze(1))

        return x_masked, mask
    
    def forward(self, x=None, out_rec=False,M_Ratio=0.75):

        if out_rec:
            latent, mask = self.forward_encoder(x, M_Ratio)
            x_ = self. forward_decoder(latent)
            return latent, mask, x_
        else: 
            latent = self.forward_(x)
            return latent

    def forward_encoder(self, x=None, M_Ratio=0.75):
        pad_len = math.ceil(x.shape[2]/ self.patch_size[0])*self.patch_size[0]-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        B,C,L = x.shape
        x_masked, mask = self.random_masking(x,M_Ratio)
        feature_m = self.encoder(x_masked).permute([0,2,1])
        feature_m = self.pos_embed(feature_m,1)
        # append cls token
        cls_tokens = self.cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)
        for encoder in self.mamba:
            feature_m = encoder(feature_m)
        latent  = feature_m

        return latent, mask
    
    def forward_decoder(self, latent=None):
        feature_ = latent[:, 1:, :]
        x_ = self.decoder_pred(feature_)
        return x_

    def forward_(self, x=None,M_Ratio=0):
        pad_len = math.ceil(x.shape[2]/ self.patch_size[0])*self.patch_size[0]-x.shape[2]
        x = F.pad(x,(0,pad_len),'constant',0)
        B,C,L = x.shape
        if M_Ratio!= 0:
            x, mask = self.random_masking2(x,M_Ratio)
        feature = self.encoder(x)#.permute([0,2,1])
        return feature


# from options.train_options import TrainOptions   
# if __name__ == "__main__":
#     opt = TrainOptions().parse()
#     model = MRIMambaMAE(opt).cuda()
#     input_data = torch.randn(1, 1, 80,100,83).cuda()  # Batch size 1, 1 channel, 84x101x87 volume
#     output,_,x_ = model(input_data, out_rec=True)
#     print(output.shape)  # Should output (1, num_classes)

# from options.train_options import TrainOptions   
# if __name__ == "__main__":
#     opt = TrainOptions().parse()
#     opt.max_position_embeddings =15000
#     model = GeneMambaMAE(opt).cuda()
#     snp = torch.randint(0, 4, (3231148,)).cuda().unsqueeze(0)
#     snp_onehot = F.one_hot(snp.clamp(0,2),num_classes=3)
#     snp_onehot[snp==3] = 0
#     output,_,x_ = model(snp_onehot.permute([0,2,1]).float(), out_rec=True)
#     print(output.shape) 
#     print(x_.shape) 
