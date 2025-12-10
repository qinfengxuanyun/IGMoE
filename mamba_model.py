import math
import torch
from torch.nn import functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
# from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from mamba_ssm import Mamba
from functools import partial

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        position = torch.arange(config.max_position_embeddings).unsqueeze(1)    
        div_term = torch.exp(
            torch.arange(0, config.hidden_size, 2) * (-math.log(10000.0) / config.hidden_size)
        )                                                
        pe = torch.zeros(config.max_position_embeddings, config.hidden_size) 
        pe[:, 0::2] = torch.sin(position * div_term)   
        pe[:, 1::2] = torch.cos(position * div_term)          
        self.position_embeddings = nn.Parameter(pe)  #pe.cuda()# 
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings2 = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.emb_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size
    
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = torch.cat([self.word_embeddings(input_ids),inputs_embeds],dim=1)
        elif input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        # token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # embeddings = inputs_embeds + token_type_embeddings
        # position_embeddings = self.position_embeddings(position_ids)
        # position_embeddings2 = self.position_embeddings2(position_ids)
        # embeddings = position_embeddings * embeddings + position_embeddings2    

        embeddings = inputs_embeds
        position_embeddings = self.position_embeddings[:seq_length].unsqueeze(0).to(embeddings.device)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings =  position_embeddings2*embeddings + position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings,inputs_embeds

class Embeddings2(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings2 = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.emb_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size
    
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = torch.cat([self.word_embeddings(input_ids),inputs_embeds],dim=1)
        elif input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = position_embeddings * embeddings + position_embeddings2    

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings,inputs_embeds

class Embeddings3(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings2 = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.emb_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size
        self.fc= nn.Linear(config.hidden_size,config.hidden_size)
    
    def apply_rope(self, x):
        bsz, seqlen, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

        # 生成频率张量
        half_dim = head_dim // 2
        theta = 10000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(x.device)
        seq_idx = torch.arange(seqlen, dtype=torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # [seq_len, half_dim]

        # 转换为 cos/sin 编码
        sin = freqs.sin()[None, :, :]  # shape: [1, seq_len, half_dim]
        cos = freqs.cos()[None, :, :]

        # 拆分并旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = torch.cat([self.word_embeddings(input_ids),inputs_embeds],dim=1)
        elif input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        # token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # embeddings = inputs_embeds + token_type_embeddings
        # position_embeddings = self.position_embeddings(position_ids)
        # position_embeddings2 = self.position_embeddings2(position_ids)
        # embeddings = position_embeddings * embeddings + position_embeddings2    

        embeddings =  self.fc(self.apply_rope(inputs_embeds))
        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings,inputs_embeds

class ClassificationHead4(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_num=512,feature_num=128):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size,32)
        self.dense2 = nn.Linear(config.hidden_size,feature_num)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(feature_num, config.num_labels)
        self.pooling = nn.AdaptiveMaxPool2d((1,config.hidden_size))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.config = config
        self.male_fc = nn.Linear(2,config.hidden_size*2)
        self.norm = nn.GroupNorm(4,config.hidden_size)
        self.nolin = nn.ELU()
          
    def forward(self, x=None, attention_mask2=None, age_sex=None):
        #x = features[:,0,:] 
        x = x * attention_mask2.unsqueeze(2) 
        x = self.pooling(x).squeeze(1)    
        # x = self.dropout1(x)   

        x = self.dense2(x)
        feature = gelu_new(x)

        if age_sex is not None:
            style = self.nolin(self.male_fc(age_sex))
            feature = (1.0+style[:,0:512]) *  self.norm(feature) + style[:,512:] #style[:,512:]+0*feature#

        x = self.dropout2(feature)
        x = self.out_proj(x)
        return x,feature

class ClassificationHead3(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_num=512,feature_num=128):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size,32)
        self.dense2 = nn.Linear(config.hidden_size,feature_num)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(feature_num, config.num_labels)
        self.pooling = nn.AdaptiveMaxPool2d((1,config.hidden_size))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.config = config

    def forward(self, x, attention_mask2, **kwargs):
        #x = features[:,0,:] 
        # x = x * attention_mask2.unsqueeze(2) 
        x = self.pooling(x).squeeze(1)    
        x = self.dropout1(x)   

        x = self.dense2(x)
        feature = gelu_new(x)

        x = self.dropout2(feature)
        x = self.out_proj(x)
        return x,feature

class ClassificationHead2(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_num=512,feature_num=128):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size,32)
        self.dense2 = nn.Linear(config.hidden_size,feature_num)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(feature_num, config.num_labels)
        self.pooling = nn.AdaptiveMaxPool2d((1,config.hidden_size))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.config = config

    def forward(self, x, attention_mask2, **kwargs):
        x = self.dropout1(x)   

        x = self.dense2(x)
        feature = gelu_new(x)

        x = self.dropout2(feature)
        x = self.out_proj(x)
        return x,feature

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        feature_num = config.hidden_size
        self.dense1 = nn.Linear(config.hidden_size,32)
        self.dense2 = nn.Linear(config.hidden_size,feature_num)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(feature_num, config.block_size*config.num_labels) #4331 3399
        self.pooling = nn.AdaptiveMaxPool2d((1,config.hidden_size))#nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.config = config

    def forward(self, features, attention_mask2, out_rec,**kwargs):
        #x = features[:,0,:] 
        x = features * attention_mask2.unsqueeze(2) 
        x = self.pooling(x).squeeze(1)    
        x = self.dropout1(x)   

        feature = self.dense2(x)
        # feature = gelu_new(feature)
        
        if out_rec:
            x = self.out_proj(feature)
            x = x.view(x.shape[0],self.config.block_size,-1) #3399
            return x,feature
        else:
            return feature

class BiMamba_block(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.encoder =Mamba(
            d_model=config.hidden_size, # Model dimension d_model
            d_state=16,  # SSM state expansion factor #16
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ) 
        self.encoder2 =Mamba(
            d_model=config.hidden_size, # Model dimension d_model
            d_state=16,  # SSM state expansion factor #16
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ) 
        self.out =  nn.Linear(config.hidden_size*2,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)#

    def forward(self, hidden_states):
        hidden_states_f = self.encoder(hidden_states)
        hidden_states_b =torch.flip(self.encoder2(torch.flip(hidden_states,dims=[1])),dims=[1])
        # hidden_states_fb =self.out(torch.cat([hidden_states_f,hidden_states_b],dim=2))
        # hidden_states = self.LayerNorm(self.dropout(hidden_states_fb)+ hidden_states)
        hidden_states = self.LayerNorm(self.dropout(hidden_states_f+hidden_states_b)+ hidden_states)
        return hidden_states
    
class Mamba_block(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.encoder =Mamba(
            d_model=config.hidden_size, # Model dimension d_model
            d_state=16,  # SSM state expansion factor #16
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ) 
        self.out =  nn.Linear(config.hidden_size*2,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)#

    def forward(self, hidden_states):
        hidden_states_f = self.encoder(hidden_states)
        hidden_states = self.LayerNorm(self.dropout(hidden_states_f)+ hidden_states)
        return hidden_states

class Mamba_dflow_MAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings2(config) 
        self.encoder =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.classifier = ClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        labels =None,
        position_ids=None,
        w_sl = None,
        width = None,
        batch = None,
        gate =None,
        inputs_embeds=None,
        output_embedding = False,
        out_rec=True
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
            #raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if gate is not None:
            if gate.shape[0]==batch_size:
                gate_mask = gate
            else:
                gate = gate.unsqueeze(0).repeat(batch_size,1)
                gate_mask = gate
        else:
            if w_sl.shape[0]==batch_size:
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
            else:
                w_sl = w_sl.unsqueeze(0).repeat(batch_size,1,1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
        
        embedding_output,inputs_embeds_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids = position_ids
        )

        # if output_embedding:
        #     embedding_output = torch.autograd.Variable(embedding_output,requires_grad=True)

        sequence_output = gate.unsqueeze(2) *embedding_output
        for encoder in self.encoder:
            sequence_output = encoder(sequence_output)
            
        if out_rec:
            logits,feature = self.classifier(sequence_output,gate_mask,out_rec)
        else:
            feature = self.classifier(sequence_output,gate_mask,out_rec)
            logits = None

        if output_embedding:
            return logits, gate, feature, inputs_embeds_output
        else:
            return logits, gate, feature

class Mamba_dflow(nn.Module):
    def __init__(self, config=None,feature_num=512):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config) 
        self.encoder =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.classifier = ClassificationHead4(config,feature_num=feature_num)
        self.male_fc = nn.Linear(2,config.hidden_size*2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)#nn.GroupNorm(4,config.hidden_size)
        self.nolin = nn.ELU()
        # mamba init

    def forward(
        self,
        gate =None,
        inputs_embeds=None,
        w_sl = None,
        input_ids=None,
        width = None,
        batch = None,
        output_embedding = False,
        use_embedding = True,
        out_seq=False,
        age_sex = None,
        train = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
            #raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if gate is not None:
            if gate.shape[0]==batch_size:
                gate_mask = gate
            else:
                gate = gate.unsqueeze(0).repeat(batch_size,1)
                gate_mask = gate
        else:
            if train:
                if w_sl.shape[0]==batch_size:
                    gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                    gate_mask = gate
                else:
                    w_sl = w_sl.unsqueeze(0).repeat(batch_size,1,1)
                    gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                    gate_mask = gate
            else:
                if w_sl.shape[0]==batch_size:
                    gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = True)[:,:,0]
                    gate_mask = gate
                else:
                    w_sl = w_sl.unsqueeze(0).repeat(batch_size,1,1)
                    gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = True)[:,:,0]
                    gate_mask = gate
        
        if use_embedding:
            embedding_output,inputs_embeds_output = self.embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds
            )
        else:
            embedding_output = inputs_embeds

        # if output_embedding:
        #     embedding_output = torch.autograd.Variable(embedding_output,requires_grad=True)
            
        # if age_sex is not None:
        #     style = self.nolin(self.male_fc(age_sex)).unsqueeze(1)
        #     embedding_output = (1.0+style[:,:,0:512]) *  self.norm(embedding_output) + style[:,:,512:] 

        sequence_output = gate.unsqueeze(2) * embedding_output
        for encoder in self.encoder:
            sequence_output = gate.unsqueeze(2) * encoder(sequence_output)
        
        if out_seq:
            if output_embedding:
                return gate, sequence_output, embedding_output
            else:
                return gate, sequence_output
        else:
            logits,feature = self.classifier(sequence_output,gate_mask,age_sex)
            if output_embedding:
                return logits, gate, feature, embedding_output
            else:
                return logits, gate, feature

class Mamba_dflow_v2(nn.Module):
    def __init__(self, config=None,feature_num=512):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings2(config) 
        self.encoder =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.classifier = ClassificationHead4(config,feature_num=feature_num)
        self.male_fc = nn.Linear(2,config.hidden_size*2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)#nn.GroupNorm(4,config.hidden_size)
        self.nolin = nn.ELU()
        # mamba init

    def forward(
        self,
        input_ids=None,
        w_sl = None,
        width = None,
        batch = None,
        gate =None,
        inputs_embeds=None,
        output_embedding = False,
        use_embedding = True,
        out_seq=False,
        age_sex = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
            #raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if gate is not None:
            if gate.shape[0]==batch_size:
                gate_mask = gate
            else:
                gate = gate.unsqueeze(0).repeat(batch_size,1)
                gate_mask = gate
        else:
            if w_sl.shape[0]==batch_size:
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
            else:
                w_sl = w_sl.unsqueeze(0).repeat(batch_size,1,1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
        
        if use_embedding:
            embedding_output,inputs_embeds_output = self.embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds
            )
        else:
            embedding_output = inputs_embeds

        # if output_embedding:
        #     embedding_output = torch.autograd.Variable(embedding_output,requires_grad=True)
            
        # if age_sex is not None:
        #     style = self.nolin(self.male_fc(age_sex)).unsqueeze(1)
        #     embedding_output = (1.0+style[:,:,0:512]) *  self.norm(embedding_output) + style[:,:,512:] 

        sequence_output = gate.unsqueeze(2) * embedding_output
        for encoder in self.encoder:
            sequence_output = gate.unsqueeze(2) * encoder(sequence_output)
        
        if out_seq:
            if output_embedding:
                return gate, sequence_output, embedding_output
            else:
                return gate, sequence_output
        else:
            logits,feature = self.classifier(sequence_output,gate_mask,age_sex)
            if output_embedding:
                return logits, gate, feature, embedding_output
            else:
                return logits, gate, feature

class Mamba_dflow_v3(nn.Module):
    def __init__(self, config=None,feature_num=512):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings3(config) 
        self.encoder =  nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.classifier = ClassificationHead4(config,feature_num=feature_num)
        self.male_fc = nn.Linear(2,config.hidden_size*2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)#nn.GroupNorm(4,config.hidden_size)
        self.nolin = nn.ELU()
        # mamba init

    def forward(
        self,
        input_ids=None,
        w_sl = None,
        width = None,
        batch = None,
        gate =None,
        inputs_embeds=None,
        output_embedding = False,
        use_embedding = True,
        out_seq=False,
        age_sex = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
            #raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if gate is not None:
            if gate.shape[0]==batch_size:
                gate_mask = gate
            else:
                gate = gate.unsqueeze(0).repeat(batch_size,1)
                gate_mask = gate
        else:
            if w_sl.shape[0]==batch_size:
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
            else:
                w_sl = w_sl.unsqueeze(0).repeat(batch_size,1,1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
        
        if use_embedding:
            embedding_output,inputs_embeds_output = self.embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds
            )
        else:
            embedding_output = inputs_embeds

        # if output_embedding:
        #     embedding_output = torch.autograd.Variable(embedding_output,requires_grad=True)
            
        # if age_sex is not None:
        #     style = self.nolin(self.male_fc(age_sex)).unsqueeze(1)
        #     embedding_output = (1.0+style[:,:,0:512]) *  self.norm(embedding_output) + style[:,:,512:] 

        sequence_output = gate.unsqueeze(2) * embedding_output
        for encoder in self.encoder:
            sequence_output = gate.unsqueeze(2) * encoder(sequence_output)
        
        if out_seq:
            if output_embedding:
                return gate, sequence_output, embedding_output
            else:
                return gate, sequence_output
        else:
            logits,feature = self.classifier(sequence_output,gate_mask,age_sex)
            if output_embedding:
                return logits, gate, feature, embedding_output
            else:
                return logits, gate, feature

class Mamba_sflow(nn.Module):
    def __init__(self, config=None,feature_num=512):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config) 
        self.encoder =  nn.ModuleList([
            Mamba_block(config) for i in range(config.num_hidden_layers)
            ])
        self.classifier = ClassificationHead4(config,feature_num=feature_num)

    def forward(
        self,
        input_ids=None,
        w_sl = None,
        width = None,
        batch = None,
        gate =None,
        age_sex = None,
        inputs_embeds=None,
        output_embedding = False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0],input_ids.size()[1]+inputs_embeds.size()[1])
            #raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if gate is not None:
            if gate.shape[0]==batch_size:
                gate_mask = gate
            else:
                gate = gate.unsqueeze(0).repeat(batch_size,1)
                gate_mask = gate
        else:
            if w_sl.shape[0]==batch_size:
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10), tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
            else:
                w_sl = w_sl.unsqueeze(0).repeat(batch_size,1,1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl,1.0-w_sl],dim=2)+1e-10), tau = 1, dim = 2, hard = False)[:,:,0]
                # gate = F.gumbel_softmax(torch.cat([w_sl,1.0-w_sl],dim=2)*10, tau = 1, dim = 2, hard = False)[:,:,0]
                gate_mask = gate
        
        embedding_output,inputs_embeds_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds
        )

        sequence_output = gate.unsqueeze(2) * embedding_output
        for encoder in self.encoder:
            sequence_output = gate.unsqueeze(2) * encoder(sequence_output)
        
        if age_sex is not None:
            logits,feature = self.classifier(sequence_output,gate_mask,age_sex)
        else:
            logits,feature = self.classifier(sequence_output,gate_mask)
        if output_embedding:
            return logits, gate, feature, inputs_embeds_output
        else:
            return logits, gate, feature
