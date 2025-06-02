import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import transformers as ppb
from torchcrf import CRF
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from rational_model.metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

class Bert_grus(nn.Module):
    def __init__(self, args):
        super(Bert_grus, self).__init__()
        # self.lay=False
        self.args = args
        # if args.cell_type=='bert':
        self.gen_encoder=ppb.BertModel.from_pretrained('bert-base-uncased')
            #self.pred_encoder = Rationale_Bert.from_pretrained('bert-base-uncased')
        # elif args.cell_type=='electra':
        #     self.gen_encoder = ppb.ElectraModel.from_pretrained("google/electra-small-discriminator")
        #     self.pred_encoder = Rationale_Electra.from_pretrained("google/electra-small-discriminator")
        # elif args.cell_type=='roberta':
        #     self.gen_encoder = ppb.RobertaModel.from_pretrained("roberta-base")
        #     self.pred_encoder = Rationale_Roberta.from_pretrained("roberta-base")
        # else:
        #     print('not defined model type')
        self.z_dim = 2
        self.dropout = nn.Dropout(args.dropout)

        freeze_para=['embeddings']              #freeze the word embedding

        for name,para in self.gen_encoder.named_parameters():
            for ele in freeze_para:
                if ele in name:
                    para.requires_grad=False
                    print('freeze the generator embedding')
        if args.freeze_bert==1:                 #固定整个bert
            for name, para in self.gen_encoder.named_parameters():
                para.requires_grad = False

        # print('numlayer:',args.num_layers)
        # print('layernorm2={}'.format(self.lay))
        self.gen = nn.GRU(input_size=768, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        # self.gen_linear = nn.Linear(768, self.z_dim)
        self.gen_linear = nn.Linear(1024, self.z_dim)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        #z = torch.where(torch.tensor(rationale_logits) != 0, torch.tensor(1), torch.tensor(0))
        torch.manual_seed(12252018)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks, segment_mask):
        masks_ = masks.unsqueeze(-1)
        gen_bert_out=self.gen_encoder(inputs, token_type_ids=segment_mask)[0]
        enc = self.dropout(gen_bert_out)
        gen_logits,_ = self.gen(enc)
        # output_logits = gen_logits.mean(dim=-1)
        output_logits = self.gen_linear(gen_logits)     

        rationale = self.independent_straight_through_sampling(output_logits) # (batch_size, seq_length, 2)
        unrationale = 1 - rationale

        # z_add_special_tokens=torch.max(rationale[:,:,1],special_masks)     #cls和sep恒为1
        # print(rationale)

        return rationale[:,:,1], unrationale[:,:,1]

    def loss(self, inputs, masks, special_masks, classifier, device, sparsity_lambda, sparsity_percentage, continuity_lambda):
        gen_bert_out = self.gen_encoder(inputs, masks)[0]
        enc = self.dropout(gen_bert_out)
        gen_layernorm = self.layernormcrf(enc)
        max_len = gen_layernorm.size()[1]
        #gen_layernorm = gen_layernorm.transpose(0,1)
        masks = masks.bool()

        gen_logits=self.crf.decode(gen_layernorm,masks)
        gen_logits[0] = gen_logits[0] + [0] * (max_len - len(gen_logits[0]))
        gen_logits= pad_sequence([torch.tensor(lst) for lst in gen_logits],batch_first=True)
        gen_tmp_logits=self.crf(gen_layernorm,gen_logits.to(device),masks)

        #gen_logits = self.generator_linear(gen_bert_out)

        ########## Sample ##########
        rationale_z = self.independent_straight_through_sampling(gen_logits).float().to(device)   # (batch_size, seq_length, 2)

        z_add_special_tokens = torch.max(rationale_z, special_masks)  # cls和sep恒为1
        sparsity_loss = sparsity_lambda * get_sparsity_loss(
            rationale_z, masks, sparsity_percentage)

        continuity_loss = continuity_lambda * get_continuity_loss(
            rationale_z)
        with torch.no_grad():
            full_text_logits=classifier.train_one_step(inputs, masks, special_masks)
            forward_logit=classifier.train_one_step(inputs, rationale_z, special_masks)
        jsd_loss = F.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        return sparsity_loss + continuity_loss + jsd_loss



