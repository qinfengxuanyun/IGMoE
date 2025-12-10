#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ScaleDense import ScaleDense_VAE,GeneResEncoder,GeneMLPEncoder
from t1f_gene_clip_ukb_rec_dataset import  MRIandGenedataset,GroupedBatchSampler
from options.train_options import TrainOptions
from mamba_model import Mamba_sflow,Mamba_dflow_MAE,Mamba_dflow
from vit import MRIMambaMAE,GeneMambaMAE
from moe import MoEMTL,MoEMTL2
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

    def forward(self, image_features=None, snp_features=None, mask=None, age_sex=None,group=False,label=None):
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
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)
       
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

tasks = ["reaction_time","symbol_digit","trail_making"]#,"age"]#tower
task = opt.task#

# pretrain_dir =  f"./generation_models/UKB_T1-GENE-CLIP_MAEREC_Mask_0.8_0.8_v1"
# ep = 300
pretrain_dir =  f"./generation_models/UKB_T1-GENE-CLIP_MAEREC_Mask_0.8_0.8_v2"
ep = 90
MODEL_PATH = f"./generation_models/UKB_T1-GENE-CLIP_MAEREC_Mask_{opt.mask_dropout}_{opt.mask_dropout2}_v4" #_Mask_{opt.mask_dropout}_{opt.mask_dropout2}
LOG_PATH = f"./logs/log_ukb_t1-gene-clip_MAEREC_Mask_{opt.mask_dropout}_{opt.mask_dropout2}_v4" 
#log_ukb_t1-gene-clip_MAEREC_Mask_0.8_0.8_v2
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
SNP_NUM =  3231148#3233344#
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
    E = Mamba_dflow(opt).cuda()#Mamba_dflow_MoE(opt).cuda()#BigBird3(opt).cuda()
    E2 = Mamba_dflow(opt2).cuda()#Mamba_dflow_MoE(opt).cuda()#BigBird3(opt).cuda()
    G  = MRIMambaMAE(opt).cuda()  #ScaleDense_VAE().cuda() #
    G2  =  GeneMLPEncoder(input_channels=PATCH*3).cuda() #GeneMambaMAE(opt2).cuda() #
    model = CLIP().cuda()
    # C = MoEMTL2(opt,num_experts=16,keep_experts=6,task_types=['rec','rec','rec','rec','rec','rec']).cuda()
    C = MoEMTL(opt,task_types=['rec','rec','rec','rec','rec','rec']).cuda()

    G = nn.DataParallel(G)
    G2 = nn.DataParallel(G2)
    E = nn.DataParallel(E)
    E2 = nn.DataParallel(E2)
    C = nn.DataParallel(C)
      
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
    # C.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_C_{ep}.pth")) #57 24
    model.load_state_dict(torch.load(pretrain_dir+f"/fold_{fold}_clip_{ep}.pth")) #57 24

    optim = torch.optim.AdamW([{'params': model.parameters()}], lr=lr)
    optim_E = torch.optim.AdamW([{'params': E.parameters()},{'params': E2.parameters()},{'params': G2.parameters()}], lr=lr)
    optim_C = torch.optim.AdamW([{'params': C.parameters()}], lr=lr) #{'params': criterion_mse_bl.parameters()}
    scaler = GradScaler()

    for ep in range(EPOCH):
        start_time = time.time()
        print(f'epoch {ep+1}')
        total_clip_loss = 0
        total_mse_loss = 0
        total_mse_loss1 = 0
        total_mse_loss2 = 0
        E.train()
        E2.train()
        G.eval()
        G2.train()
        C.train()
        C.module.training = True#C.module.moe.training = True
        model.train() #3231148
        for  train_data in data_loader_train:
            fid, feature, label, fc_range,age_sex, integer_encoded, scores  = train_data
  
            label = label.cuda()
            B, L = integer_encoded.shape
            feature = feature.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda().long()
            scores = scores.cuda()

            # with autocast(dtype=torch.bfloat16):
            w_sl_use = torch.ones_like(w_sl).cuda() * torch.bernoulli(torch.full_like(w_sl,opt.mask_dropout)).unsqueeze(0).repeat(B,1,1) #opt.mask_dropout
            # w_sl_use2 = torch.ones_like(w_sl2).cuda() * torch.bernoulli(torch.full_like(w_sl2,opt.mask_dropout2)).unsqueeze(0).repeat(B,1,1)
            # w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
            w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B,1,1)
            gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use,1.0-w_sl_use],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
            gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2,1.0-w_sl_use2],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]

            snp_onehot = F.one_hot(snp.clamp(0,2),num_classes=3)
            snp_onehot[snp==3] = 0
            feature2,mask_snp =  G2(snp_onehot.permute([0,2,1]).float(),M_Ratio=1.0-opt.mask_dropout2,p=PATCH)

            _,_, mri_feature, feature_emb = E(gate = gate,inputs_embeds=feature,output_embedding=True) 
            _,_, snp_feature, feature2_emb = E2(gate = gate2,inputs_embeds=feature2,output_embedding=True,age_sex=age_sex) 
        
            loss_mse_list = []
            loss_mse1_list = []
            loss_mse2_list = []
            mode = random.choice([1, 2])
            for task_index, task in enumerate(tasks):
                if task != 'age':
                    score_index= (scores[:,task_index]!=-100.0) & (label[:,task_index]!=-1)
                    true_sample_num = torch.sum(score_index.int())
                    if true_sample_num == 0:
                        continue

                    y1,y2,y,_ = C(mri_feature,snp_feature,task_index,mode)
                    loss_mse_task  = criterion_mse(y[score_index,0], scores[score_index,task_index])# 
                    loss_mse1_task  = criterion_mse(y1[score_index,0],scores[score_index,task_index])# -label[score_index,task_index]
                    loss_mse2_task  = criterion_mse(y2[score_index,0], scores[score_index,task_index])# 
                else:
                    y1,y2,y,_ = C(mri_feature,snp_feature,task_index,mode)
                    loss_mse_task  =  torch.tensor(0.0,requires_grad=True).to(age_sex.device)
                    loss_mse1_task  = 10*criterion_mse(y1[:,0]-age_sex[:,0], y2[:,0].detach()-age_sex[:,0])# -label[score_index,task_index]
                    loss_mse2_task  = 10*criterion_mse(y2[:,0], age_sex[:,0])# 

                loss_mse_list.append(loss_mse_task)
                loss_mse1_list.append(loss_mse1_task)
                loss_mse2_list.append(loss_mse2_task)
            loss_mse =  sum(loss_mse_list)
            loss_mse1 =  sum(loss_mse1_list)
            loss_mse2 =  sum(loss_mse2_list)

            mask = torch.ones((B,B)).cuda()
            loss_clip,_,mri_feature,snp_feature = model(mri_feature, snp_feature, mask,None,False)
        
            loss = loss_mse + loss_mse1+ loss_mse2 + loss_clip 

            optim.zero_grad()
            optim_E.zero_grad()
            optim_C.zero_grad()
            loss.backward()
            optim.step()
            optim_E.step()
            optim_C.step()
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.step(optim_E)
            # scaler.step(optim_C)
            # scaler.update()
            total_mse_loss += loss_mse.item()   
            total_mse_loss1 += loss_mse1.item()   
            total_mse_loss2 += loss_mse2.item()   
            total_clip_loss += loss_clip.item()
                
        print("mse loss:", np.round(total_mse_loss/dataset_size,5),
              "mse loss1:",np.round(total_mse_loss1/dataset_size,5),
              "mse loss2:",np.round(total_mse_loss2/dataset_size,5),
              "clip loss:", np.round(total_clip_loss/dataset_size,5))
        writer.add_scalars('mse_loss', {'mse_loss' + str(fold): total_mse_loss/dataset_size, }, ep+1) 
        writer.add_scalars('mse_loss1', {'mse_loss' + str(fold): total_mse_loss1/dataset_size, }, ep+1)      
        writer.add_scalars('mse_loss2', {'mse_loss' + str(fold): total_mse_loss2/dataset_size, }, ep+1)           
        writer.add_scalars('clip_loss', {'clip_loss' + str(fold): total_clip_loss/dataset_size, }, ep+1)      

        ###########test phase############################   
        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        C.eval()
        C.module.training = False#C.module.moe.training = False
        model.eval()
        with torch.no_grad():
            pre_all = [[[] for i in range(3)] for task in tasks]
            label_all = [[[] for i in range(3)] for task in tasks]
            mri_feature_list = []
            snp_feature_list = []
            age_sex_list = []
            for  test_data in data_loader_test:
                fid, feature, label, fc_range,age_sex, integer_encoded, scores  = test_data

                label = label.cuda()
                B, L = integer_encoded.shape
                feature = feature.cuda()
                snp = integer_encoded.cuda().long()
                age_sex = age_sex.cuda()
                scores = scores.cuda()

                w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B,1,1)
                w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B,1,1)

                snp_onehot = F.one_hot(snp.clamp(0,2),num_classes=3)
                snp_onehot[snp==3] = 0
                feature2,mask_snp =  G2(snp_onehot.permute([0,2,1]).float(),p=PATCH)

                _,gate, mri_feature = E(w_sl = w_sl_use,inputs_embeds=feature)#,out_seq=True) 
                _,gate2, snp_feature = E2(w_sl = w_sl_use2,inputs_embeds=feature2,age_sex=age_sex)#,out_seq=True) 

                mode = 2
                for task_index, task in enumerate(tasks):
                    if task != 'age':
                        score_index= (scores[:,task_index]!=-100.0)  & (label[:,task_index]!=-1)
                        true_sample_num = torch.sum(score_index.int())
                        if true_sample_num == 0:
                            continue

                        y1,y2,y,_ = C(mri_feature,snp_feature,task_index,mode)

                        for pre_index, pre in enumerate([y1,y2,y]):
                            pre_all[task_index][pre_index].extend(pre[score_index,0].cpu().numpy().tolist())
                            label_all[task_index][pre_index].extend(scores[score_index,task_index].cpu().numpy().tolist())
                    else:
                        y1,y2,y,_ = C(mri_feature,snp_feature,task_index,mode)

                        for pre_index, pre in enumerate([y1,y2,y]):
                            if pre_index!= 0:
                                pre_all[task_index][pre_index].extend(pre[:,0].cpu().numpy().tolist())
                                label_all[task_index][pre_index].extend(age_sex[:,0].cpu().numpy().tolist())
                            else:
                                pre_all[task_index][pre_index].extend((pre[:,0]-age_sex[:,0]).cpu().numpy().tolist())
                                label_all[task_index][pre_index].extend((y2[:,0]-age_sex[:,0]).cpu().numpy().tolist())

                mri_feature,snp_feature = model.forward2(mri_feature, snp_feature,None)
                mri_feature_list.append(mri_feature.cpu())
                snp_feature_list.append(snp_feature.cpu())
                age_sex_list.append(age_sex.cpu())    

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
              
        mri_features = torch.cat(mri_feature_list,dim=0)
        snp_features = torch.cat(snp_feature_list,dim=0)
        sub_num = mri_features.shape[0]

        age_sexs = torch.cat(age_sex_list,dim=0)
        ages = age_sexs[:, 0].unsqueeze(1)
        sexs = age_sexs[:, 1].unsqueeze(1)
        mask = ((sexs == sexs.T) & (torch.abs(ages - ages.T) <= 0)).float()

        mri_features = mri_features / mri_features.norm(dim=1, keepdim=True)
        snp_features = snp_features / snp_features.norm(dim=1, keepdim=True)
        
        clip_scores = (snp_features @ mri_features.t() * mask).numpy()
        top10_SR = diagonal_topNum_probability(M=clip_scores,k=10)
        top5_SR = diagonal_topNum_probability(M=clip_scores,k=5)
        top1_SR = diagonal_topNum_probability(M=clip_scores,k=1)#!/usr/bin/env python

        clip_scores2 = (mri_features @ snp_features.t() * mask).numpy()
        top10_MR = diagonal_topNum_probability(M=clip_scores2,k=10)
        top5_MR = diagonal_topNum_probability(M=clip_scores2,k=5)
        top1_MR = diagonal_topNum_probability(M=clip_scores2,k=1)

        print(f"ukb_test_top10_SR:",np.round(top10_SR,5),
              f"ukb_test_top5_SR:",np.round(top5_SR,5),
              f"ukb_test_top1_SR:",np.round(top1_SR,5))
        print(f"ukb_test_top10_MR:",np.round(top10_MR,5),
              f"ukb_test_top5_MR:",np.round(top5_MR,5),
              f"ukb_test_top1_MR:",np.round(top1_MR,5))
        writer.add_scalars(f'ukb_test_top10_SR', {'test_top10_SR_' + str(fold): top10_SR, }, ep+1) 
        writer.add_scalars(f'ukb_test_top5_SR', {'test_top5_SR_' + str(fold): top5_SR, }, ep+1)
        writer.add_scalars(f'ukb_test_top1_SR', {'test_top1_SR_' + str(fold): top1_SR, }, ep+1)
        writer.add_scalars(f'ukb_test_top10_MR', {'test_top10_MR_' + str(fold): top10_MR, }, ep+1) 
        writer.add_scalars(f'ukb_test_top5_MR', {'test_top5_MR_' + str(fold): top5_MR, }, ep+1)
        writer.add_scalars(f'ukb_test_top1_MR', {'test_top1_MR_' + str(fold): top1_MR, }, ep+1)

        end_time = time.time()
        print(f"time: {end_time-start_time}s")

        if ((ep+1) % 5) == 0:
            torch.save(model.state_dict(), MODEL_PATH + f"/fold_{fold}_clip_{ep+1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_clip_{ep+1}.pth")
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
