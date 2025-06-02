import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from rational_model.metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math
import torch.distributed as dist
from sklearn import metrics
import sys
sys.path.append('../')

from rational_model.metric import compute_micro_stats


def reduce_value(value,world_size=2,average=False):
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value=value/world_size

        return value



class JS_DIV(nn.Module):
    def __init__(self):
        super(JS_DIV, self).__init__()
        self.kl_div=nn.KLDivLoss(reduction='batchmean',log_target=True)
    def forward(self,p,q):
        p=F.softmax(p,dim=-1)
        q=F.softmax(q,dim=-1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl_div(m, p.log()) + self.kl_div(m, q.log()))




def ram_set(intensor):
    tensor_tmp  = copy.deepcopy(intensor)
    ones_positions = (tensor_tmp == 1).nonzero(as_tuple=True)

    # 计算值为1的总数量
    num_ones = ones_positions[0].size(0)

    # 随机选择一半的位置
    half_num_ones = int(num_ones // 1.8)
    random_indices = torch.randperm(num_ones)[:half_num_ones]

    for i in random_indices:
        row_index = ones_positions[0][i]
        col_index = ones_positions[1][i]
        tensor_tmp[row_index, col_index] = 0

    # 选择前一半的位置
    # num_ones = ones_positions[0].size(0)

    # # 将前一半的位置的值设置为0
    # for i in range(half_num_ones,num_ones):
    #     row_index = ones_positions[0][i]
    #     col_index = ones_positions[1][i]
    #     tensor_tmp[row_index, col_index] = 0

    return tensor_tmp



def validate_share_bert_onegpu(model, classifier, annotation_loader, device,opt,flag):
    num_predicted_pos = 0.
    num_words = 0

    targets_all, outputs_all, outputs_full_all = None, None, None
    n_test_correct, n_test_to_model_correct, n_test_total = 0, 0, 0 
    result = []
    ori_text = []
    ori_polarity = []
    ori_aspect = []
    ori_pred = []
    ori_full_pred = []
    model.eval()
    with torch.no_grad():
        for (batch, sample_batched) in enumerate(annotation_loader):

            # inputs, masks, labels, annotations,special_masks = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
                # device), special_masks.to(device)
            inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
            # text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, edge_adj, src_mask, aspect_mask = inputs
            # text_bert_indices = sample_batched['concat_bert_indices'].to(opt.device)
            text_bert_indices = sample_batched['concat_bert_indices'].to(opt.device)
            attention_mask = sample_batched['attention_mask'].to(opt.device) 
            labels = sample_batched['polarity'].to(opt.device)
            sp_mask = sample_batched['special_mask'].to(opt.device)
            concat_segments_indices = sample_batched['concat_segments_indices'].to(opt.device)
            attention_mask = attention_mask
            # rationales -- (batch_size, seq_length, 2)
            full_mask = torch.max(attention_mask,sp_mask)
            rationales, rationales_add_special_token = model(text_bert_indices, full_mask, concat_segments_indices)
            # rationales = ram_set(attention_mask)

            rantion_mask_mid = torch.mul(rationales , attention_mask)
            rantion_mask = torch.max(rantion_mask_mid, sp_mask) 

            _, num_predicted_pos_, _ = compute_micro_stats(
                rantion_mask_mid, rantion_mask_mid)#预测对的，预测的，真正的
            classifier.eval()
            with torch.no_grad():
                cls_full_logits =classifier(inputs, full_mask)    
            cls_logits =classifier(inputs, rantion_mask)
            pred = torch.argmax(cls_logits, dim=-1)
            pred_full = torch.argmax(cls_full_logits, dim=-1)


            # # TP predict 和 label 同时为1
            # TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # # TN predict 和 label 同时为0
            # TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # # FN predict 0 label 1
            # FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # # FP predict 1 label 0
            # FP += ((pred == 1) & (labels == 0)).cpu().sum()

            targets_all = torch.cat((targets_all, labels), dim=0) if targets_all is not None else labels
            outputs_all = torch.cat((outputs_all, pred), dim=0) if outputs_all is not None else pred
            outputs_full_all = torch.cat((outputs_full_all, pred_full), dim=0) if outputs_full_all is not None else pred
            n_test_correct += (pred == labels).sum().item()
            n_test_to_model_correct += (pred == pred_full).sum().item()
            n_test_total += len(pred)
            # num_true_pos += num_true_pos_
            # sp_pos_ = torch.sum(sp_mask)
            num_predicted_pos = num_predicted_pos + num_predicted_pos_ #- sp_pos_
            
            # num_real_pos += num_real_pos_
            num_words = num_words + torch.sum(attention_mask)

            #结果写入
            ori_text.extend(sample_batched['text'])
            ori_aspect.extend(sample_batched['term'])
            ori_polarity.extend(sample_batched['polarity'].tolist())
            ori_pred.extend(pred.cpu().tolist())
            ori_full_pred.extend(pred_full.cpu().tolist())

            result.extend(rantion_mask.detach().cpu().numpy())

        #rationales预测结果
        # micro_precision = num_true_pos / num_predicted_pos#预测对的占预测的多少
        # micro_recall = num_true_pos / num_real_pos#预测对的占对的多少
        # micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                        #    micro_recall)
        sparsity = num_predicted_pos / num_words#预测的占总的多少

        # cls
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f1_score = 2 * recall * precision / (recall + precision)
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_score = metrics.f1_score(targets_all.cpu(), outputs_all.cpu(), labels=[0, 1, 2], average='macro')
        accuracy = n_test_correct / n_test_total
        accuracy_to_full_model = n_test_to_model_correct / n_test_total
        f1_score_to_full_model = metrics.f1_score(outputs_full_all.cpu(), outputs_all.cpu(), labels=[0, 1, 2], average='macro')
        print(
            "val dataset : f1-score:{:.4f} accuracy:{:.4f} accuracy_to_model:{:.4f} f1-score_to_model:{:.4f}".format( f1_score, accuracy, accuracy_to_full_model, f1_score_to_full_model))
        # print('sp',sparsity)
        # a = hh
        if flag == True:
            # import time
            report = metrics.classification_report(outputs_full_all.cpu(), outputs_all.cpu(), digits=4)
            confusion = metrics.confusion_matrix(outputs_full_all.cpu(), outputs_all.cpu())
            from time import strftime, localtime
            # np_array = rantion_mask.detach().cpu().numpy()

            # answer = text_bert_indices.detach().cpu().numpy()
            # text = sample_batched['text']
            file_path = './absa/rantional_result/{}_{}_{}-{}.txt'.format(opt.model_name,opt.dataset,opt.sparsity_percentage,strftime("%Y-%m-%d_%H:%M:%S", localtime()))
            with open(file_path, 'w') as file:
                # 写入文本内容
                for i in range(0,len(ori_text)):
                    file.write(ori_text[i] + '\n')
                    file.write(ori_aspect[i] + '\n')
                    file.write(str(ori_polarity[i]) + '\n')
                    file.write(str(ori_pred[i]) + '\n')
                    file.write(str(ori_full_pred[i]) + '\n')
                    # file.write(' '.join(map(str, result[i])) + '\n')
                    ones_position = []
                    for j, item in enumerate(result[i]):
                        if item == 1.0:
                            ones_position.append(j)
                    file.write(' '.join(map(str, ones_position)) + '\n')
                # file.write('您可以在这里写入任意文本内容。')
            # a = hh   
            return report, confusion, sparsity, accuracy, f1_score, accuracy_to_full_model, f1_score_to_full_model
    return sparsity, accuracy, f1_score, accuracy_to_full_model

def train_bert_bcr_onegpu_distillation(model,classifier, opt_gen, dataset, device, opt,annotation_loader,best_acc,best_model):
    targets_all, outputs_all, ouputs_full_all = None, None, None
    n_test_correct, n_test_total,n_model_correct = 0, 0, 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    model.train()
    data_size=len(dataset)
    check_size=int(data_size/4)
    for batch, sample_batched in enumerate(dataset):
        model.train()
        opt_gen.zero_grad()
        #opt_pred.zero_grad()
        inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
        # select_label = sample_batched['select_len'].to(opt.device).float()
        # text_bert_indices = sample_batched['concat_bert_indices'].to(opt.device) 
        text_bert_indices = sample_batched['concat_bert_indices'].to(opt.device) 
        concat_segments_indices = sample_batched['concat_segments_indices'].to(opt.device)
        attention_mask = sample_batched['attention_mask'].to(opt.device)
        labels = sample_batched['polarity'].to(opt.device)
        select_label = sample_batched['select_len'].to(opt.device).float()
        select_label = select_label+2
        sp_mask = sample_batched['special_mask'].to(opt.device)
        # inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        full_mask = torch.max(attention_mask,sp_mask)

        rationales, unrationales = model(text_bert_indices, full_mask, concat_segments_indices)

        unrationale_mask_mid = torch.mul(unrationales , attention_mask)
        unrationale_mask = torch.max(unrationale_mask_mid, sp_mask) 
        rantion_mask_mid = torch.mul(rationales , attention_mask)
        rantion_mask = torch.max(rantion_mask_mid, sp_mask) 
        # rantion_mask = rantion_mask_mid

        sparsity_loss = opt.sparsity_lambda * get_sparsity_loss(rantion_mask_mid, attention_mask, opt.sparsity_percentage)
        # rantion_mask_mid_count = torch.sum(rantion_mask_mid, dim=1)
        # sparsity_loss = torch.sum(torch.abs(rantion_mask_mid_count-select_label))

        continuity_loss = opt.continuity_lambda * get_continuity_loss(rantion_mask_mid)


        
        #full_text_logits select_text_logits
        # classifier.model.eval()
        forward_logit = classifier(inputs, rantion_mask)
        with torch.no_grad():
            full_text_logits = classifier(inputs, full_mask)
        un_forward_logit = classifier(inputs, rational_mask=unrationale_mask)

        # rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        #forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)

        #jsd
        if opt.div == 'js':
            jsd_func = JS_DIV()
            # jsd_loss = jsd_func(forward_logit, full_text_logits)
            # un_jsd_loss = jsd_func(un_forward_logit, full_text_logits)
        elif opt.div == 'kl':#


            jsd_loss = F.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1), reduction='batchmean')
            un_jsd_loss = F.kl_div(F.softmax(un_forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1), reduction='batchmean')
        else:
            print('div wrong')

        gen_loss = sparsity_loss +  0.1*continuity_loss + jsd_loss + 1/un_jsd_loss
        # gen_loss =  sparsity_loss + + 10*jsd_loss + 10/un_jsd_loss
        # gen_loss = un_jsd_loss#+jsd_loss #- 10*un_jsd_loss

        # print("output requires_grad:", forward_logit.requires_grad)  # 应该是 True

        with open("./train_loss.txt", encoding="utf-8",mode="a") as file:  
            file.write('new:  ')
            file.write(str(gen_loss))
            file.write('\n')

        # print("gen_loss requires_grad:", gen_loss.requires_grad)
       # gen_loss =  model.loss(inputs, masks, special_masks, classifier, device, args.sparsity_lambda, args.sparsity_percentage, args.continuity_lambda)#nputs, masks, special_masks, classifier, device, sparsity_lambda, sparsity_percentage, continuity_lambda):
        #gen_loss = sparsity_loss + continuity_loss + abs(full_text_cls_loss)
        # update gradient
        # gen_loss.requires_grad_(True)
        gen_loss.backward()#！！！！！！
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
                   
        opt_gen.step()

        
        # for name, parms in model.gen.named_parameters():	
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print("===")
        #     # break
        # a = hh
        opt_gen.zero_grad()



        pred = torch.argmax(forward_logit, dim=-1)

        full_labels = torch.argmax(full_text_logits, dim=-1)
        # print(pred)
        # TP predict 和 label 同时为1
        # TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # # TN predict 和 label 同时为0 
        # TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # # FN predict 0 label 1
        # FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # # FP predict 1 label 0
        # FP += ((pred == 1) & (labels == 0)).cpu().sum()
        #cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

        targets_all = torch.cat((targets_all, labels), dim=0) if targets_all is not None else labels
        ouputs_full_all = torch.cat((ouputs_full_all, full_labels), dim=0) if ouputs_full_all is not None else full_labels
        outputs_all = torch.cat((outputs_all, pred), dim=0) if outputs_all is not None else pred

        n_test_correct += (pred == labels).sum().item()
        n_model_correct += (pred == full_labels).sum().item()
        n_test_total += len(pred)

        if (batch+1)%check_size==0:

            print("check with {}".format(batch))
            annotation_results = validate_share_bert_onegpu(model, classifier, annotation_loader, device,opt,False)
            print('jsd_loss', jsd_loss,'\n continuity_loss', continuity_loss,"\n sparsity_loss",sparsity_loss,"\n gen_loss",gen_loss)
            print("The val performance: sparsity: %.2f,accuracy: %2f,f1_score :%2f,full_accuracy: %2f" % (100 * annotation_results[0], 100 * annotation_results[1], 100 * annotation_results[2], 100 * annotation_results[3]))
            # print(
            #     "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            #     % (100 * annotation_results[0], 100 * annotation_results[1],

            if annotation_results[3]>best_acc:
                if annotation_results[0] < 0.5:
                    best_model = copy.deepcopy(model)
                    best_acc = annotation_results[3]




    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # f1_score = 2 * recall * precision / (recall + precision)
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # precision = 0
    # recall = 0
    f1_score = metrics.f1_score(targets_all.cpu(),outputs_all.cpu(), labels=[0, 1, 2], average='macro')
    accuracy = n_test_correct / n_test_total
    accuracy_to_full_model = n_model_correct / n_test_total


    return f1_score, accuracy, accuracy_to_full_model, best_model, best_acc



