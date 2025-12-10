#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ScaleDense import ScaleDense_VAE,GeneResEncoder,GeneMLPEncoder
from t1f_gene_clip_ukb_rec_dataset import  MRIandGenedataset,GroupedBatchSampler
from options.train_options import TrainOptions
from mamba_model import Mamba_sflow,Mamba_dflow_MAE,Mamba_dflow
from vit import MRIMambaMAE,GeneMambaMAE
from moe import MoEMTL, MoEMTL2, MaskcomputeMoE, CLIPMoE, MaskcomputeMoE3
import itertools
import copy
import time
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
from torch.cuda.amp import autocast, GradScaler
import random
import warnings

warnings.filterwarnings("ignore")
cuda = torch.cuda.is_available()

def diagonal_topNum_probability(M=None,k=5):
    """
    判断矩阵 M 中每行的对角线元素是否位于该行前 5 大元素之列，并返回其所占比例。

    参数：
    M: 形状为 (N, N) 的 numpy 数组

    返回：
    prob: 对角线元素位于各自行前 k 大值的概率
    """
    N = M.shape[0]
    count_topk = 0

    for i in range(N):
        row = M[i]
        diagonal_val = row[i]

        threshold_kth = np.sort(row)[-k]  
        if diagonal_val >= threshold_kth:
            count_topk += 1

    prob = count_topk / N
    return prob

def diagonal_topNum_probability2(M=None,labels=None,k=5):
    """
    判断矩阵 M 中每行的对角线元素是否位于该行前 5 大元素之列，并返回其所占比例。

    参数：
    M: 形状为 (N, N) 的 numpy 数组

    返回：
    prob: 对角线元素位于各自行前 k 大值的概率
    """
    N = M.shape[0]
    count_topk = 0

    for i in range(N):
        row = M[i]
        index_k = np.argsort(row)[-k:]  
        for index in index_k:
            if labels[index] == labels[i]:
                count_topk += 1
                # break
    prob = count_topk / N / k
    return prob

class CLIP(nn.Module):
    def __init__(self, temperature=0.07,feature_num=512):
        super(CLIP, self).__init__()
        self.image_proj = nn.Linear(feature_num, 512)
        self.snp_proj = nn.Linear(feature_num, 512)
        self.male_fc = nn.Linear(2,1024)
        self.image_pooling = nn.AdaptiveMaxPool2d((1,feature_num))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.snp_pooling = nn.AdaptiveMaxPool2d((1,feature_num))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.nolin = nn.ELU()
        self.norm = nn.GroupNorm(4,feature_num)

    def forward(self, image_features=None, snp_features=None, mask=None, age_sex=None,group=False):
        # image_features = self.image_pooling(image_features)
        image_features = image_features.view(image_features.size(0), -1)   
        # style = self.nolin(self.male_fc(age_sex))
        # image_features = (1.0+style[:,0:512]) *  self.norm(image_features) + style[:,512:]
        image_features = self.image_proj(image_features)
       
        # snp_features = self.snp_pooling(snp_features)
        snp_features = snp_features.view(snp_features.size(0), -1)
        snp_features = self.snp_proj(snp_features)

        clip_scores = F.cosine_similarity(image_features, snp_features)
        # normalized features
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ snp_features_norm.t()

        if group:
            loss_img = torch.sum(-1.0 * F.log_softmax(logits, dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss_snp = torch.sum(-1.0 * F.log_softmax(logits.t(), dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss = loss_img + loss_snp
        else:
            # 计算对角线元素（正样本）的索引
            labels = torch.arange(logits.shape[0], device=logits.device)
            # 计算损失
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

        return loss / 2,clip_scores,image_features,snp_features
    
    def forward2(self, image_features=None, snp_features=None,age_sex=None):
        # image_features = self.image_pooling(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        # style = self.nolin(self.male_fc(age_sex))
        # image_features = (1.0+style[:,0:512]) *  self.norm(image_features) + style[:,512:]
        image_features = self.image_proj(image_features)
       
        # snp_features = self.snp_pooling(snp_features)
        snp_features = snp_features.view(snp_features.size(0), -1)
        snp_features = self.snp_proj(snp_features)
        return image_features,snp_features
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class BalancedMSELoss(nn.Module):
    def __init__(self, init_noise_sigma = 1.0):
        super().__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        logits = -(pred.unsqueeze(1)-target.unsqueeze(1).T).pow(2) / (2 *noise_var)
        # B = pred.shape[0]
        # mask = torch.zeros((B,B),device=pred.device)
        # for i in range(B):
        #     for j in range(B):
        #         if torch.abs(target[i] - target[j]) < 0.01:
        #             mask[i,j] = 1
        # loss = torch.sum(-1.0 * F.log_softmax(logits, dim=1) * mask  / (torch.sum(mask,dim=1,keepdim=True)+1e-12) )
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0],device=pred.device))
        loss = loss * (2 * noise_var.detach())
        return loss
    
criterion = nn.CrossEntropyLoss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_mse_bl = BalancedMSELoss().cuda()
criterion_mse_bl2 = BalancedMSELoss().cuda()
criterion_mse_bl3 = BalancedMSELoss().cuda()

opt = TrainOptions().parse()
# initial for recurrence
seed = 8
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

tasks =["reaction_time","symbol_digit","trail_making"]#tower
task = opt.task#
sparse_flag = 1
sparse_flag2 = 1

ep = 100
pretrain_dir =  f"./generation_models/UKB_T1-GENE-CLIP_MAEREC_Mask_0.8_0.8_v4"
# ep = 20
# pretrain_dir  =  f"./generation_models/UKB_T1-GENE-CLIP_MAEREC_Mask_v4_{'-'.join(tasks)}_{sparse_flag}_{sparse_flag2}_32-6_64-8_f"

MODEL_PATH = f"./generation_models/UKB_T1-GENE-CLIP_MAEREC_Mask_v4_{'-'.join(tasks)}_{sparse_flag}_{sparse_flag2}_{opt.mri_num_experts}-{opt.mri_keep_experts}_{opt.snp_num_experts}-{opt.snp_keep_experts}_f_5" #_Mask_{opt.mask_dropout}_{opt.mask_dropout2}
LOG_PATH = f"./logs/log_ukb_t1-gene-clip_MAEREC_Mask_v4_{'-'.join(tasks)}_{sparse_flag}_{sparse_flag2}_{opt.mri_num_experts}-{opt.mri_keep_experts}_{opt.snp_num_experts}-{opt.snp_keep_experts}_f_5" 

os.system("mkdir -p {}".format(MODEL_PATH))
os.system("mkdir -p {}".format(LOG_PATH))

TRAIN_BATCH_SIZE = 8#32
TEST_BATCH_SIZE = 8#32
lr = 1e-4
EPOCH = 500
WORKERS = TRAIN_BATCH_SIZE
WIDTH = 32
NUM = 10*13*11#1200#1200 #
NUM2 = 6311#6316#12622#12622#
PATCH = 512
EPOCH_PRE= 5
SNP_NUM = 3231148 #3233344#
MRI_TOPK = opt.mri_th
SNP_TOPK = opt.snp_th
writer = SummaryWriter(logdir=LOG_PATH, comment='Gene2MRI')

for fold in range(5):
    if fold!=1:
        continue
    dataset_train = MRIandGenedataset(fold=fold,phase="train")
    # data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
    #                                                 num_workers=WORKERS,drop_last=True,pin_memory=True)
    sampler = GroupedBatchSampler(dataset_train, TRAIN_BATCH_SIZE)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler,num_workers=WORKERS,pin_memory=True)
    
    dataset_test = MRIandGenedataset(fold=fold,phase="test")
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                    num_workers=WORKERS,drop_last=True,pin_memory=True)
    
    dataset_size = len(data_loader_train)
    dataset_size_test = len(data_loader_test)
    print("dataset_size: ", dataset_size)
    print("data_loader_test: ", dataset_size_test)

    # opt.hidden_dropout_prob = 0.2
    # opt.emb_dropout_prob = 0.2
    opt.classifier_dropout = 0
    opt2 = copy.deepcopy(opt)
    opt2.max_position_embeddings = 30000#
    # opt2.hidden_size = 1024
    # opt2.num_hidden_layers = 3
    E = Mamba_dflow(opt).cuda()#
    E2 = Mamba_dflow(opt2).cuda()#
    G  = MRIMambaMAE(opt).cuda() #ScaleDense_VAE().cuda() #
    G2  =  GeneMLPEncoder(input_channels=PATCH*3).cuda() #GeneMambaMAE(opt2).cuda() #
    model = CLIP().cuda()
    # C = MoEMTL2(opt,num_experts=16,keep_experts=6,task_types=['rec','rec','rec','rec','rec','rec']).cuda()
    C = MoEMTL(opt,task_types=['rec','rec','rec','rec','rec','rec']).cuda()
    M1 = MaskcomputeMoE(opt,token_num=NUM,num_experts=opt.mri_num_experts,keep_experts=opt.mri_keep_experts).cuda()
    M2 = MaskcomputeMoE(opt2,token_num=NUM2,num_experts=opt.snp_num_experts,keep_experts=opt.snp_keep_experts).cuda()
    modelMoE = CLIPMoE().cuda()
    MoEE = Mamba_dflow(opt).cuda()#Mamba_dflow_MoE(opt).cuda()#BigBird3(opt).cuda()
    MoEE2 = Mamba_dflow(opt2).cuda()#Mamba_dflow_MoE(opt).cuda()#BigBird3(opt).cuda()

    G = nn.DataParallel(G)
    G2 = nn.DataParallel(G2)
    E = nn.DataParallel(E)
    E2 = nn.DataParallel(E2)
    C = nn.DataParallel(C)
    M1 = nn.DataParallel(M1)
    M2 = nn.DataParallel(M2)
    MoEE = nn.DataParallel(MoEE)
    MoEE2 = nn.DataParallel(MoEE2)
      
    for p in G.parameters():
        p.requires_grad = False
    for p in G2.parameters():
        p.requires_grad = True
    for p in E.parameters():
        p.requires_grad = True
    for p in E2.parameters():
        p.requires_grad = True
    for p in model.parameters():
        p.requires_grad = True
    for p in C.parameters():
        p.requires_grad = True
    for p in M1.parameters():
        p.requires_grad = True
    for p in M2.parameters():
        p.requires_grad = True
    for p in modelMoE.parameters():
        p.requires_grad = True
    for p in MoEE.parameters():
        p.requires_grad = True
    for p in MoEE2.parameters():
        p.requires_grad = True
    w_sl = torch.ones([NUM,1])
    w_sl = w_sl.cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.ones([NUM2,1])
    w_sl2 = w_sl2.cuda()
    w_sl2.requires_grad = False

    G.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_G_{ep}.pth")) #,strict=False
    G2.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_G2_{ep}.pth")) #57 24
    E.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_E_{ep}.pth")) #100
    E2.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_E2_{ep}.pth")) #57 24
    C.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_C_{ep}.pth")) #57 24
    model.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_clip_{ep}.pth")) #57 24
    # modelMoE.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_clipMoE_{ep}.pth")) #57 24
    # M1.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_M1_{ep}.pth")) #100
    # M2.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_M2_{ep}.pth")) #57 24
    # MoEE.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_E_{ep}.pth")) #100
    # MoEE2.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_E2_{ep}.pth")) #57 24

    # optim = torch.optim.AdamW([{'params': model.parameters()}], lr=lr) 
    optim = torch.optim.AdamW([{'params': model.parameters()},{'params': modelMoE.parameters()}], lr=lr)
    optim_E = torch.optim.AdamW([{'params': E.parameters()},{'params': E2.parameters()},
                                #  {'params': MoEE.parameters()},{'params': MoEE2.parameters()},
                                 {'params': G2.parameters()},], lr=lr)
    optim_C = torch.optim.AdamW([{'params': C.parameters()}], lr=lr) 
    optim_M = torch.optim.AdamW([{'params': M1.parameters()},{'params': M2.parameters()}], lr=lr)
    scaler = GradScaler()
    
    for ep in range(EPOCH):
        start_time = time.time()
        print(f'epoch {ep+1}')
        total_clip_loss = 0
        total_mse_loss = 0
        total_mse_loss1 = 0
        total_mse_loss2 = 0
        total_mse_loss_c = 0
        total_mse_loss1_c = 0
        total_mse_loss2_c = 0
        total_clip_loss_c = 0
        total_bl_loss1= 0
        total_bl_loss2 = 0
        MoEE.train()
        MoEE2.train()
        E.train()
        E2.train()
        G.eval()
        G2.train()
        C.train()
        model.train() #3231148
        M1.train() 
        M2.train() 
        modelMoE.train()
        C.module.training = True#C.module.moe.training = True
        M1.module.training = True
        M2.module.training = True
        M1.module.epoch = ep
        M2.module.epoch = ep
        for  train_data in data_loader_train:
            fid, feature, label, fc_range,age_sex, integer_encoded, scores  = train_data
  
            label = label.cuda()
            B, L = integer_encoded.shape
            feature = feature.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda().long()
            scores = scores.cuda()
            
            with autocast(dtype=torch.bfloat16):
                w_sl_use = torch.ones_like(w_sl).cuda() * torch.bernoulli(torch.full_like(w_sl,opt.mask_dropout)).unsqueeze(0).repeat(B,1,1) #opt.mask_dropout
                # w_sl_use2 = torch.ones_like(w_sl2).cuda() * torch.bernoulli(torch.full_like(w_sl2,opt.mask_dropout2)).unsqueeze(0).repeat(B,1,1)
                # w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
                w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B,1,1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use,1.0-w_sl_use],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2,1.0-w_sl_use2],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]

                snp_onehot = F.one_hot(snp.clamp(0,2),num_classes=3)
                snp_onehot[snp==3] = 0
                feature2,mask_snp =  G2(snp_onehot.permute([0,2,1]).float(),M_Ratio=1.0-opt.mask_dropout2,p=PATCH)
                
                feature_emb,_ = E.module.embeddings(inputs_embeds=feature)
                feature2_emb,_ = E2.module.embeddings(inputs_embeds=feature2)
                _,_, mri_feature = E(gate = gate,inputs_embeds=feature_emb, use_embedding = False)
                _,_, snp_feature = E2(gate = gate2,inputs_embeds=feature2_emb,age_sex=age_sex, use_embedding = False)

                # loss_mse_list = []
                # loss_mse1_list = []
                # loss_mse2_list = []
                # for task_index, task in enumerate(tasks):
                #     task_index = tasks.index(task)
                #     score_index= (scores[:,task_index]!=-100.0) & (label[:,task_index]!=-1)
                #     true_sample_num = torch.sum(score_index.int())
                #     if true_sample_num == 0:
                #         continue
                    
                #     mode = random.choice([1, 2])
                #     y1,y2,y,_ = C(mri_feature,snp_feature,task_index,mode)

                #     loss_mse_task  = criterion_mse(y[score_index,0], scores[score_index,task_index])# 
                #     loss_mse1_task  = criterion_mse(y1[score_index,0],scores[score_index,task_index])# -label[score_index,task_index]
                #     loss_mse2_task  = criterion_mse(y2[score_index,0], scores[score_index,task_index])# 
                #     loss_mse_list.append(loss_mse_task)
                #     loss_mse1_list.append(loss_mse1_task)
                #     loss_mse2_list.append(loss_mse2_task)

                # loss_mse =  sum(loss_mse_list)
                # loss_mse1 =  sum(loss_mse1_list)
                # loss_mse2 =  sum(loss_mse2_list)

                # mask = torch.ones((B,B)).cuda()
                # loss_clip,_,mri_feature_s,snp_feature_s = model(mri_feature, snp_feature, mask,None,False)

                ###########################################################
                if sparse_flag:
                    # expert_outputs,_ = M1(input_features=feature_emb, mode="share")
                    mri_G,mri_I,loss_bl1 = M1(input_features=feature_emb, mode="share")
                    loss_bl1 = torch.mean(loss_bl1)
                if sparse_flag2:
                    # expert_outputs2,_ = M2(input_features=feature2_emb, mode="share")
                    snp_G,snp_I,loss_bl2 = M2(input_features=feature2_emb, mode="share")
                    loss_bl2 = torch.mean(loss_bl2)

                loss_mse_c_list = []
                loss_mse1_c_list = []
                loss_mse2_c_list = []
                loss_clip_c_list = []
                for task_index, task in enumerate(tasks):
                    task_index = tasks.index(task)
                    score_index= (scores[:,task_index]!=-100.0) & (label[:,task_index]!=-1)
                    true_sample_num = torch.sum(score_index.int())
                    if true_sample_num == 0:
                        continue
                    
                    if sparse_flag:
                        feature_emb_c,_,_ = M1(I=mri_I, G=mri_G, input_features=feature_emb, input_features2=mri_feature,task_index=task_index, mode="task")
                        # feature_emb_c,_ = M1(expert_outputs=expert_outputs, input_features2=mri_feature,task_index=task_index, mode="task")
                    if sparse_flag2:
                        feature2_emb_c,_,_ = M2(I=snp_I, G=snp_G, input_features=feature2_emb, input_features2=snp_feature,task_index=task_index, mode="task")
                        # feature2_emb_c,_ = M2(expert_outputs=expert_outputs2, input_features2=snp_feature,task_index=task_index, mode="task")

                    _,_, mri_feature_c = E(gate = gate, inputs_embeds=feature_emb_c,batch = B, use_embedding = False)        
                    _,_, snp_feature_c = E2(gate = gate2, inputs_embeds=feature2_emb_c,batch = B,age_sex=age_sex, use_embedding = False)         

                    mode = random.choice([1, 2])
                    y1_c,y2_c,y_c,_ = C(mri_feature_c,snp_feature_c,task_index,mode)
                    
                    loss_mse_c_task  = criterion_mse(y_c[score_index,0], scores[score_index,task_index])# 
                    loss_mse1_c_task  = criterion_mse(y1_c[score_index,0],scores[score_index,task_index])# -label[score_index,task_index]
                    loss_mse2_c_task  = criterion_mse(y2_c[score_index,0], scores[score_index,task_index])# 
                    loss_mse_c_list.append(loss_mse_c_task)
                    loss_mse1_c_list.append(loss_mse1_c_task)
                    loss_mse2_c_list.append(loss_mse2_c_task)

                    mask = torch.ones((B,B)).cuda()
                    loss_clip_c_task,_,_,_ = modelMoE(mri_feature_c, snp_feature_c, mask,None,False,task_index)
                    loss_clip_c_list.append(loss_clip_c_task)

                loss_mse_c=  sum(loss_mse_c_list)
                loss_mse1_c =  sum(loss_mse1_c_list)
                loss_mse2_c =  sum(loss_mse2_c_list)
                loss_clip_c =  sum(loss_clip_c_list)
                
                ###########################################################
                # loss = loss_mse + loss_mse1+ loss_mse2 + loss_clip
                # loss +=  0.1*(loss_mse_c+ loss_mse1_c+ loss_mse2_c + loss_clip_c) #   
                loss = loss_mse_c+ loss_mse1_c+ loss_mse2_c + (loss_bl1+loss_bl2)  #+ loss_clip_c #

            optim.zero_grad()
            optim_E.zero_grad()
            optim_C.zero_grad()
            optim_M.zero_grad()
            # loss.backward()
            # optim.step()
            # optim_E.step()
            # optim_C.step()
            # optim_M.step()
            scaler.scale(loss).backward()
            # scaler.step(optim)
            scaler.step(optim_E)
            scaler.step(optim_C)
            if sparse_flag or sparse_flag2:
                scaler.step(optim_M)
            scaler.update()
            # total_mse_loss += loss_mse.item()   
            # total_mse_loss1 += loss_mse1.item()   
            # total_mse_loss2 += loss_mse2.item()   
            # total_clip_loss += loss_clip.item()
            total_mse_loss_c += loss_mse_c.item()   
            total_mse_loss1_c += loss_mse1_c.item()   
            total_mse_loss2_c += loss_mse2_c.item()   
            total_clip_loss_c += loss_clip_c.item()
            total_bl_loss1 += loss_bl1.item()
            total_bl_loss2 += loss_bl2.item()
                
        # print("mse loss:", np.round(total_mse_loss/dataset_size,5),
        #       "mse loss1:",np.round(total_mse_loss1/dataset_size,5),
        #       "mse loss2:",np.round(total_mse_loss2/dataset_size,5),
        #       "clip loss:", np.round(total_clip_loss/dataset_size,5))
        print("mse loss c:", np.round(total_mse_loss_c/dataset_size,5),
              "mse loss1 c:",np.round(total_mse_loss1_c/dataset_size,5),
              "mse loss2 c:",np.round(total_mse_loss2_c/dataset_size,5),
              "clip loss c:",np.round(total_clip_loss_c/dataset_size,5),
              "bl loss1:",np.round(total_bl_loss1/dataset_size,5),
              "bl loss2:",np.round(total_bl_loss2/dataset_size,5),
              )
        
        writer.add_scalars('mse_loss', {'mse_loss' + str(fold): total_mse_loss/dataset_size, }, ep+1) 
        writer.add_scalars('mse_loss1', {'mse_loss' + str(fold): total_mse_loss1/dataset_size, }, ep+1)      
        writer.add_scalars('mse_loss2', {'mse_loss' + str(fold): total_mse_loss2/dataset_size, }, ep+1)    
        writer.add_scalars('clip_loss', {'clip_loss' + str(fold): total_clip_loss/dataset_size, }, ep+1)         
        writer.add_scalars('mse_loss_c', {'mse_loss' + str(fold): total_mse_loss_c/dataset_size, }, ep+1) 
        writer.add_scalars('mse_loss1_c', {'mse_loss' + str(fold): total_mse_loss1_c/dataset_size, }, ep+1)      
        writer.add_scalars('mse_loss2_c', {'mse_loss' + str(fold): total_mse_loss2_c/dataset_size, }, ep+1)     
        writer.add_scalars('clip_loss_c', {'clip_loss' + str(fold): total_clip_loss_c/dataset_size, }, ep+1)   
        writer.add_scalars('bl_loss1', {'bl_loss' + str(fold): total_bl_loss1/dataset_size, }, ep+1)      
        writer.add_scalars('bl_loss2', {'bl_loss' + str(fold): total_bl_loss2/dataset_size, }, ep+1)   

        ###########test phase############################   
        MoEE.eval()
        MoEE2.eval()
        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        C.eval()
        model.eval()
        M1.eval() 
        M2.eval() 
        modelMoE.eval()
        C.module.training = False#C.module.moe.training = False
        M1.module.training = False
        M2.module.training = False
        with torch.no_grad():
            total_mask1 = [torch.zeros([NUM]).cuda() for task in tasks]
            total_mask2 = [torch.zeros([NUM2]).cuda() for task in tasks]
            total_experts_mask1 = [torch.zeros([M1.module.num_experts]).cuda() for task in tasks]
            total_experts_mask2 = [torch.zeros([M2.module.num_experts]).cuda() for task in tasks]
            for  train_data in data_loader_train:
                fid, feature, label, fc_range,age_sex, integer_encoded, scores  = train_data
                B, L = integer_encoded.shape
                feature = feature.cuda()
                snp = integer_encoded.cuda().long()
                scores = scores.cuda()
                label = label.cuda()

                w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
                w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B,1,1)

                snp_onehot = F.one_hot(snp.clamp(0,2),num_classes=3)
                snp_onehot[snp==3] = 0
                feature2,mask_snp =  G2(snp_onehot.permute([0,2,1]).float(),p=PATCH)

                feature_emb,_ = E.module.embeddings(inputs_embeds=feature)
                feature2_emb,_ = E2.module.embeddings(inputs_embeds=feature2)
                _,_, mri_feature = E(gate = gate,inputs_embeds=feature_emb, use_embedding = False)
                _,_, snp_feature = E2(gate = gate2,inputs_embeds=feature2_emb,age_sex=age_sex, use_embedding = False)

                # ###########################################################
                if sparse_flag:
                    mri_G,mri_I = M1(input_features=feature_emb, mode="share")
                if sparse_flag2:
                    snp_G,snp_I = M2(input_features=feature2_emb, mode="share")
                for task_index, task in enumerate(tasks):
                    task_index = tasks.index(task)
                    score_index= (scores[:,task_index]!=-100.0) & (label[:,task_index]!=-1)
                    true_sample_num = torch.sum(score_index.int())
                    if true_sample_num == 0:
                        continue
                    if sparse_flag:
                        feature_emb_c,experts_mask1,mask1 = M1(I=mri_I, G=mri_G, input_features=feature_emb, input_features2 = mri_feature, task_index=task_index, mode="task")
                        total_mask1[task_index] += 1.0 * torch.sum(mask1,dim=0) / B #torch.mean(mask1[score_index],dim=0)
                        total_experts_mask1[task_index] += 1.0 * torch.sum(experts_mask1,dim=0)  / B #torch.mean(experts_mask1[score_index],dim=0)
                    if sparse_flag2:
                        feature2_emb_c,experts_mask2,mask2 = M2(I=snp_I, G=snp_G, input_features=feature2_emb, input_features2 = snp_feature,task_index=task_index, mode="task")
                        total_mask2[task_index] += 1.0 * torch.sum(mask2,dim=0)  / B #torch.mean(mask2[score_index],dim=0)
                        total_experts_mask2[task_index] += 1.0 * torch.sum(experts_mask2,dim=0)  / B #torch.mean(experts_mask2[score_index],dim=0)
       
        for task_index, task in enumerate(tasks):
            if sparse_flag:
                print("indices:",torch.sort(total_mask1[task_index]/dataset_size, descending=True)[1].cpu().numpy().tolist()[0:20])
                print("values:", [round(value,2) for value in torch.sort(total_mask1[task_index]/dataset_size, descending=True)[0].cpu().numpy().tolist()[0:20]])
                print("experts indices:",torch.sort(total_experts_mask1[task_index]/dataset_size, descending=True)[1].cpu().numpy().tolist()[0:10])
                print("experts values:", [round(value,2) for value in torch.sort(total_experts_mask1[task_index]/dataset_size, descending=True)[0].cpu().numpy().tolist()[0:10]])
                np.save(MODEL_PATH + f"/fold_{fold}_{task}_mask1_{ep+1}.npy", (total_mask1[task_index]/dataset_size).cpu().numpy())
                np.save(MODEL_PATH + f"/fold_{fold}_{task}_expertmask1_{ep+1}.npy", (total_experts_mask1[task_index]/dataset_size).cpu().numpy()) 
            if sparse_flag2:
                print("indices2:",torch.sort(total_mask2[task_index]/dataset_size, descending=True)[1].cpu().numpy().tolist()[0:20])
                print("values2:",[round(value,2) for value in torch.sort(total_mask2[task_index]/dataset_size, descending=True)[0].cpu().numpy().tolist()[0:20]])
                print("experts2 indices:",torch.sort(total_experts_mask2[task_index]/dataset_size, descending=True)[1].cpu().numpy().tolist()[0:10])
                print("experts2 values:", [round(value,2) for value in torch.sort(total_experts_mask2[task_index]/dataset_size, descending=True)[0].cpu().numpy().tolist()[0:10]])
                np.save(MODEL_PATH + f"/fold_{fold}_{task}_mask2_{ep+1}.npy", (total_mask2[task_index]/dataset_size).cpu().numpy())        
                np.save(MODEL_PATH + f"/fold_{fold}_{task}_expertmask2_{ep+1}.npy", (total_experts_mask2[task_index]/dataset_size).cpu().numpy())
            
        #############################################################################
        with torch.no_grad():
            pre_all = [[[] for i in range(3)] for task in tasks]
            label_all = [[[] for i in range(3)] for task in tasks]
            for  test_data in data_loader_test:
                fid, feature, label, fc_range,age_sex, integer_encoded, scores  = test_data
                B, L = integer_encoded.shape
                feature = feature.cuda()
                snp = integer_encoded.cuda().long()
                scores = scores.cuda()
                label = label.cuda()

                w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
                w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B,1,1)

                snp_onehot = F.one_hot(snp.clamp(0,2),num_classes=3)
                snp_onehot[snp==3] = 0
                feature2,mask_snp =  G2(snp_onehot.permute([0,2,1]).float(),p=PATCH)

                feature_emb,_ = E.module.embeddings(inputs_embeds=feature)
                feature2_emb,_ = E2.module.embeddings(inputs_embeds=feature2)
                _,_, mri_feature = E(gate = gate,inputs_embeds=feature_emb, use_embedding = False)
                _,_, snp_feature = E2(gate = gate2,inputs_embeds=feature2_emb,age_sex=age_sex, use_embedding = False)
                ###########################################################
                if sparse_flag:
                    mri_G,mri_I = M1(input_features=feature_emb, mode="share")
                if sparse_flag2:
                    snp_G,snp_I = M2(input_features=feature2_emb, mode="share")

                for task_index, task in enumerate(tasks):
                    task_index = tasks.index(task)
                    score_index= (scores[:,task_index]!=-100.0) & (label[:,task_index]!=-1)
                    true_sample_num = torch.sum(score_index.int())
                    if true_sample_num == 0:
                        continue
                    if sparse_flag:
                        feature_emb_c,_,_ = M1(I=mri_I, G=mri_G, input_features=feature_emb, input_features2 = mri_feature,task_index=task_index, mode="task")
                    if sparse_flag2:
                        feature2_emb_c,_,_ = M2(I=snp_I, G=snp_G, input_features=feature2_emb, input_features2 = snp_feature,task_index=task_index, mode="task")

                    ###########################################################
                    _,_, mri_feature_c = E(w_sl = w_sl_use, inputs_embeds=feature_emb_c,batch = B, use_embedding = False)        
                    _,_, snp_feature_c = E2(w_sl = w_sl_use2, inputs_embeds=feature2_emb_c,batch = B,age_sex=age_sex, use_embedding = False)         

                    mode = 2
                    y1_c,y2_c,y_c,_ = C(mri_feature_c,snp_feature_c,task_index,mode)
                    for pre_index, pre in enumerate([y1_c,y2_c,y_c]):
                        pre_all[task_index][pre_index].extend(pre[score_index,0].cpu().numpy().tolist())
                        label_all[task_index][pre_index].extend(scores[score_index,task_index].cpu().numpy().tolist())
       
        for task_index, task in enumerate(tasks):
            for model_index, mode in enumerate(['snp','mri','all']):
                mae = mean_absolute_error(np.array(label_all[task_index][model_index]), np.array(pre_all[task_index][model_index]))
                rmse = mean_squared_error(np.array(label_all[task_index][model_index]), np.array(pre_all[task_index][model_index]), squared=False)
                r2 = r2_score(np.array(label_all[task_index][model_index]), np.array(pre_all[task_index][model_index]))
                print(f"test_{task}_{mode}_mae:", np.round(mae,5), 
                      f"test_{task}_{mode}_rmse:", np.round(rmse,5),
                      f"test_{task}_{mode}_r2:", np.round(r2,5))
                writer.add_scalars(f'test_{task}_{mode}_mae', {'mae': mae, }, ep+1)
                writer.add_scalars(f'test_{task}_{mode}_rmse', {'rmse': rmse, }, ep+1)
                writer.add_scalars(f'test_{task}_{mode}_r2', {'r2': r2, }, ep+1)
        
        end_time = time.time()
        print(f"time: {end_time-start_time}s")

        if ((ep+1) % 5) == 0:
            torch.save(model.state_dict(), MODEL_PATH + f"/fold_{fold}_clip_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_clip_{ep+1}.pth")
            torch.save(modelMoE.state_dict(), MODEL_PATH + f"/fold_{fold}_clipMoE_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_clipMoE_{ep+1}.pth")
            torch.save(E.state_dict(), MODEL_PATH + f"/fold_{fold}_E_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_E_{ep+1}.pth")
            torch.save(E2.state_dict(), MODEL_PATH + f"/fold_{fold}_E2_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_E2_{ep+1}.pth")
            torch.save(G.state_dict(), MODEL_PATH + f"/fold_{fold}_G_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_G_{ep+1}.pth")
            torch.save(G2.state_dict(), MODEL_PATH + f"/fold_{fold}_G2_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_G2_{ep+1}.pth")
            torch.save(C.state_dict(), MODEL_PATH + f"/fold_{fold}_C_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_C_{ep+1}.pth")
            torch.save(M1.state_dict(), MODEL_PATH + f"/fold_{fold}_M1_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_M1_{ep+1}.pth")
            torch.save(M2.state_dict(), MODEL_PATH + f"/fold_{fold}_M2_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_M2_{ep+1}.pth")
            torch.save(MoEE.state_dict(), MODEL_PATH + f"/fold_{fold}_MoEE_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_MoEE_{ep+1}.pth")
            torch.save(MoEE2.state_dict(), MODEL_PATH + f"/fold_{fold}_MoEE2_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_MoEE2_{ep+1}.pth")
    
        # if opt.use_sparse and (ep+1)>=EPOCH_PRE:
        #     sparse_flag = 1
        #     for p in M1.parameters():
        #         p.requires_grad = True
        # if opt.use_sparse2 and (ep+1)>=EPOCH_PRE:
        #     sparse_flag2 = 1
        #     for p in M2.parameters():
        #         p.requires_grad = True
