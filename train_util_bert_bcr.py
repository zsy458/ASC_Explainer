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




def train_bert_bcr(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks,special_masks)

        #full_text_logits
        full_text_logits=model.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        #jsd
        jsd_func = JS_DIV()
        jsd_loss = jsd_func(logits, full_text_logits)


        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales) / torch.sum(masks)).cpu().item()
        writer_epoch[0].add_scalar('train_sp', sparsity, writer_epoch[1]*len(dataset)+batch)
        # print(sparsity)
        # print(rationales[0,:10,1])
        train_sp.append(
            (torch.sum(rationales) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        loss = cls_loss + sparsity_loss + continuity_loss+jsd_loss+full_text_cls_loss
        # update gradient


        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])

    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy


def train_bert_bcr_multigpu(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # rationales, cls_logits
        rationales, logits = model.module(inputs, masks,special_masks)

        #full_text_logits
        full_text_logits=model.module.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        #jsd
        jsd_func = JS_DIV()
        jsd_loss = jsd_func(logits, full_text_logits)


        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales) / torch.sum(masks)).cpu().item()
        writer_epoch[0].add_scalar('train_sp', sparsity, writer_epoch[1]*len(dataset)+batch)
        # print(sparsity)
        # print(rationales[0,:10,1])
        train_sp.append(
            (torch.sum(rationales) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        loss = cls_loss + sparsity_loss + continuity_loss+jsd_loss+full_text_cls_loss
        # update gradient


        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)


    TP=reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    if args.rank==0:
        writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
        writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
        writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
        writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def train_bert_bcr_multigpu_decouple(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.module.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')



        # detach_logit = model.module.detach_gen_pred(inputs, masks, rationales_add_special_token)
        # forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)

        #full_text_logits
        full_text_logits=model.module.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = cls_loss +  full_text_cls_loss
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        elif args.gen_sparse==1:
            opt_gen.step()
            opt_gen.zero_grad()
        else:
            pass

        #train divergence
        opt_gen.zero_grad()
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.module.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.module.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.module.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False

        rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)
        full_text_logits = model.module.train_one_step(inputs, masks, special_masks)


        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')



        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)


        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.module.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.module.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.module.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)


    TP=reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    # if args.rank==0:
    #     writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def train_bert_bcr_onegpu_decouple(model, opt_gen,opt_pred, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')


        #full_text_logits
        full_text_logits=model.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = (cls_loss +  full_text_cls_loss)/2
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        else:
            pass

        #train divergence
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False

        if args.gen_acc==1:
            rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
            sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
                rationales, masks, args.sparsity_percentage)

            continuity_loss = args.continuity_lambda * get_continuity_loss(
                rationales)
        forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)
        full_text_logits = model.train_one_step(inputs, masks, special_masks)


        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')





        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()




    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print('get grad')
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # print('get grad end')
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    # if args.rank==0:
    #     writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    #     writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy

def dev_bert_bcr_multigpu(model, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    with torch.no_grad():
        for (batch, (inputs, masks, labels, special_masks)) in enumerate(dataset):


            inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
                device)

            # rationales, cls_logits


            rationales, logits = model.module(inputs, masks, special_masks)



            cls_soft_logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(cls_soft_logits, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

        # 等待所有进程
        if device!= torch.device('cpu'):
            torch.cuda.synchronize(device)
    TP = reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def dev_bert_bcr_onetigpu(model, classifier,dataset, device, opt,writer_epoch):
    # TP = 0
    # TN = 0
    # FN = 0
    # FP = 0
    cls_l = 0    
    targets_all, outputs_all = None, None
    n_test_correct, n_test_total = 0, 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    with torch.no_grad():
        for (batch,sample_batched) in enumerate(dataset):

            inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
            # text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, edge_adj, src_mask, aspect_mask = inputs
            text_bert_indices = sample_batched['concat_bert_indices'].to(opt.device)
            attention_mask = sample_batched['attention_mask'].to(opt.device)
            labels = sample_batched['polarity'].to(opt.device)
            sp_mask = sample_batched['special_mask'].to(opt.device).float()
            attention_mask =attention_mask.float()
            # inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
                # device)

            # rationales, cls_logits

            rationales, _ = model(text_bert_indices, attention_mask, attention_mask, device)
            rantion_mask_mid = torch.mul(rationales , attention_mask)
            rantion_mask = torch.max(rantion_mask_mid, sp_mask) 

            cls_logits= classifier.model(inputs,rantion_mask)


            cls_soft_logits = F.softmax(cls_logits, -1)
            _, pred = torch.max(cls_soft_logits, dim=-1)

            # TP predict 和 label 同时为1
            # TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # # TN predict 和 label 同时为0
            # TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # # FN predict 0 label 1
            # FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # # FP predict 1 label 0
            # FP += ((pred == 1) & (labels == 0)).cpu().sum()
            targets_all = torch.cat((targets_all, labels), dim=0) if targets_all is not None else labels
            outputs_all = torch.cat((outputs_all, pred), dim=0) if outputs_all is not None else pred

            n_test_correct += (pred == labels).sum().item()
            n_test_total += len(pred)



    precision = 0
    recall = 0
    f1_score = metrics.f1_score(targets_all.cpu(), outputs_all.cpu(), labels=[0, 1, 2], average='macro')
    accuracy = n_test_correct / n_test_total

    return precision, recall, f1_score, accuracy


def validate_share_bert_multigpu(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        for (batch, (inputs, masks, labels,
                     annotations,special_masks)) in enumerate(annotation_loader):
            inputs, masks, labels, annotations,special_masks = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
                device), special_masks.to(device)

            # rationales -- (batch_size, seq_length, 2)
            rationales, cls_logits = model.module(inputs, masks,special_masks)

            num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
                annotations, rationales)

            soft_pred = F.softmax(cls_logits, -1)
            _, pred = torch.max(soft_pred, dim=-1)

            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

            num_true_pos += num_true_pos_
            num_predicted_pos += num_predicted_pos_
            num_real_pos += num_real_pos_
            num_words += torch.sum(masks)

        # 等待所有进程
        if device != torch.device('cpu'):
            torch.cuda.synchronize(device)

        num_true_pos=reduce_value(num_true_pos)
        num_predicted_pos = reduce_value(num_predicted_pos)

        num_real_pos = reduce_value(num_real_pos)
        num_words = reduce_value(num_words)

        micro_precision = num_true_pos / num_predicted_pos
        micro_recall = num_true_pos / num_real_pos
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                           micro_recall)
        sparsity = num_predicted_pos / num_words

        # cls
        TP = reduce_value(TP)
        FP = reduce_value(FP)
        FN = reduce_value(FN)
        TN = reduce_value(TN)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print(
            "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                         f1_score,
                                                                                                         accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1


def ram_set(intensor):
    tensor_tmp  = copy.deepcopy(intensor)
    ones_positions = (tensor_tmp == 1).nonzero(as_tuple=True)

    # 计算值为1的总数量
    num_ones = ones_positions[0].size(0)

    # 随机选择一半的位置
    half_num_ones = num_ones // 2
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
            text_bert_indices = sample_batched['concat_bert_indices'].to(opt.device)
            attention_mask = sample_batched['attention_mask'].to(opt.device)
            labels = sample_batched['polarity'].to(opt.device)
            sp_mask = sample_batched['special_mask'].to(opt.device)
            concat_segments_indices = sample_batched['concat_segments_indices'].to(opt.device)
            attention_mask =attention_mask
            # rationales -- (batch_size, seq_length, 2)
            full_mask = torch.max(attention_mask,sp_mask)
            rationales, rationales_add_special_token = model(text_bert_indices, full_mask, concat_segments_indices, device)
            # rationales = ram_set(attention_mask)

            rantion_mask_mid = torch.mul(rationales , attention_mask)
            rantion_mask = torch.max(rantion_mask_mid, sp_mask) 

            _, num_predicted_pos_, _ = compute_micro_stats(
                rantion_mask_mid, rantion_mask_mid)#预测对的，预测的，真正的
            classifier.model.eval()
            with torch.no_grad():
                cls_logits =classifier.model(inputs, rantion_mask)
                cls_full_logits =classifier.model(inputs, full_mask)    
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

def train_bert_classifier(model,opt_pred, dataset, device, args,mode):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    if mode=='train':
        model.module.train()
    else:
        model.module.eval()
    for (batch, data) in enumerate(dataset):
        if mode=='test':
            inputs, masks, labels,annotations, special_masks=data
        else:
            inputs, masks, labels, special_masks=data
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
            device)
        if mode == 'train':
            logits=model.module.train_one_step(inputs, masks, special_masks)

            cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
            cls_loss.backward()
            opt_pred.step()
        else:
            with torch.no_grad():
                logits = model.module.train_one_step(inputs, masks, special_masks)



        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()


        # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    TP = reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def train_bert_classifier_onegpu(model,opt_pred, dataset, device, args,mode):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    if mode=='train':
        model.train()
    else:
        model.eval()
    for (batch, data) in enumerate(dataset):
        if mode=='test':
            inputs, masks, labels,annotations, special_masks=data
        else:
            inputs, masks, labels, special_masks=data
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device), special_masks.to(
            device)
        if mode == 'train':
            logits=model.train_one_step(inputs, masks, special_masks)

            cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
            cls_loss.backward()
            opt_pred.step()
        else:
            with torch.no_grad():
                logits = model.train_one_step(inputs, masks, special_masks)



        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()






    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def train_bert_bcr_multigpu_distillation(model,classifier, opt_gen,opt_pred, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []

    for (batch, (inputs, masks, labels,special_masks)) in enumerate(dataset):
        opt_gen.zero_grad()
        opt_pred.zero_grad()

        inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        if args.gen_acc == 0:
            forward_logit = model.module.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')



        # detach_logit = model.module.detach_gen_pred(inputs, masks, rationales_add_special_token)
        # forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)

        #full_text_logits
        with torch.no_grad():
            full_text_logits=classifier.train_one_step(inputs, masks,special_masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        if args.gen_sparse==1:
            classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:
            classification_loss = cls_loss +  full_text_cls_loss
        else:
            print('gen sparse wrong')

        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()

        # if args.gen_acc==1:
        #     opt_gen.step()
        #     opt_gen.zero_grad()
        # elif args.gen_sparse==1:
        #     opt_gen.step()
        #     opt_gen.zero_grad()
        # else:
        #     pass

        #train divergence
        # opt_gen.zero_grad()
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.module.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.module.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.module.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False

        # rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        forward_logit = model.module.pred_forward_logit(inputs, masks, rationales_add_special_token)



        #jsd
        if args.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div == 'kl':
            jsd_loss = nn.functional.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1),
                                            reduction='batchmean')
        else:
            print('div wrong')



        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)


        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)

        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        # update gradient

        gen_loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()
        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.module.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.module.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.module.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1





        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    # 等待所有进程
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)


    TP=reduce_value(TP)
    FP = reduce_value(FP)
    FN = reduce_value(FN)
    TN = reduce_value(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)


    return precision, recall, f1_score, accuracy


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
        text_bert_indices = sample_batched['concat_bert_indices_forex'].to(opt.device)
        concat_segments_indices = sample_batched['concat_segments_indices'].to(opt.device)
        attention_mask = sample_batched['attention_mask'].to(opt.device)
        labels = sample_batched['polarity'].to(opt.device)
        select_label = sample_batched['select_len'].to(opt.device).float()
        # select_label = select_label +2
        sp_mask = sample_batched['special_mask'].to(opt.device)
        # inputs, masks, labels, special_masks = inputs.to(device), masks.to(device), labels.to(device),special_masks.to(device)

        # train classification
        # get rationales
        full_mask = torch.max(attention_mask,sp_mask)

        rationales, unrationales = model(text_bert_indices, full_mask, concat_segments_indices, device)

        unrationale_mask_mid = torch.mul(unrationales , attention_mask)
        unrationale_mask = torch.max(unrationale_mask_mid, sp_mask) 
        rantion_mask_mid = torch.mul(rationales , attention_mask)
        rantion_mask = torch.max(rantion_mask_mid, sp_mask) 

        sparsity_loss = opt.sparsity_lambda * get_sparsity_loss(rantion_mask_mid, attention_mask, opt.sparsity_percentage)
        # rantion_mask_mid_count = torch.sum(rantion_mask_mid, dim=1)
        # sparsity_loss = torch.sum(torch.abs(rantion_mask_mid_count-select_label))

        continuity_loss = opt.continuity_lambda * get_continuity_loss(rantion_mask_mid)

        '''
        ##
        full_text_logits=classifier.train_one_step(inputs, masks,special_masks)
        select_inputs  = torch.mul(inputs,rationales_add_special_token)
        select_text_logits=classifier.train_one_step(inputs, rationales_add_special_token,special_masks)

        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, select_text_logits)
        ##
        
        if args.gen_acc == 0:#
            forward_logit = model.pred_forward_logit(inputs, masks, torch.detach(rationales_add_special_token))
        elif args.gen_acc==1:
            forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)
        else:
            print('wrong gen acc')
        '''
        
        #full_text_logits select_text_logits
        classifier.model.eval()
        with torch.no_grad():
            full_text_logits = classifier.model(inputs, rational_mask=full_mask)
            forward_logit = classifier.model(inputs, rational_mask=rantion_mask)
            un_forward_logit = classifier.model(inputs, rational_mask=unrationale_mask)

        '''
        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss=args.cls_lambda * F.cross_entropy(full_text_logits, labels)

        
        if args.gen_sparse==1:
            classification_loss=cls_loss+sparsity_loss + continuity_loss
        elif args.gen_sparse==0:#
            classification_loss = cls_loss
        else:
            print('gen sparse wrong')

        #classification_loss.backward()
        
        #opt_pred.step()
        #opt_pred.zero_grad()
        
        if args.gen_acc==1:
            opt_gen.step()
            opt_gen.zero_grad()
        elif args.gen_sparse==1:
            opt_gen.step()
            opt_gen.zero_grad()
            rationales, rationales_add_special_token = model.get_rationale(inputs, masks, special_masks)
            sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
                rationales, masks, args.sparsity_percentage)

            continuity_loss = args.continuity_lambda * get_continuity_loss(
                rationales)
        else:
            pass
        
        #train divergence
        # opt_gen.zero_grad()
        
        name1 = []
        name2 = []
        name3 = []
        for idx, p in model.pred_encoder.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx, p in model.pred_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        for idx, p in model.layernorm2.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False
        '''
        # rationales, rationales_add_special_token = model.module.get_rationale(inputs, masks, special_masks)
        #forward_logit = model.pred_forward_logit(inputs, masks, rationales_add_special_token)

        #jsd
        if opt.div == 'js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
            un_jsd_loss = jsd_func(un_forward_logit, full_text_logits)
        elif opt.div == 'kl':#
            jsd_loss = F.kl_div(F.softmax(forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1), reduction='batchmean')
            un_jsd_loss = F.kl_div(F.softmax(un_forward_logit, dim=-1).log(), F.softmax(full_text_logits, dim=-1), reduction='batchmean')
            # full_loss = F.kl_div(F.softmax(full_text_logits, dim=-1).log(), F.softmax(labels, dim=-1), reduction='batchmean')
            # full_loss = F.cross_entropy(full_text_logits, labels)
        else:
            print('div wrong')
        # print('jsd_loss:  ', jsd_loss)
        # print('un_jsd_loss:  ', un_jsd_loss)
        # print('full_loss:  ', full_loss)
        # a  = hh

        # print(sample_batched['text'])
        # print('forward_logit')
        # print(forward_logit)
        # print('un_forward_logit')
        # print(un_forward_logit)
        # print('full_text_logits')
        # print(full_text_logits)
        # print(torch.argmax(forward_logit, -1))
        # print(torch.argmax(un_forward_logit, -1))
        # print(torch.argmax(full_text_logits, -1))
        # print('labels')
        # print(labels)
        # print('sparsity_loss:',sparsity_loss)
        # print('continuity_loss:',continuity_loss)
        # print('jsd_loss:',jsd_loss)
        # print('un_jsd_loss:',un_jsd_loss)
        gen_loss = sparsity_loss +  continuity_loss + 10*jsd_loss + 0.1*(1/un_jsd_loss)
        # print('gen_loss:  ',gen_loss)
        # a = hh
        with open("./train_loss.txt", encoding="utf-8",mode="a") as file:  
            file.write('new:  ')
            file.write(str(gen_loss))
            file.write('\n')

        # print(gen_loss)

       # gen_loss =  model.loss(inputs, masks, special_masks, classifier, device, args.sparsity_lambda, args.sparsity_percentage, args.continuity_lambda)#nputs, masks, special_masks, classifier, device, sparsity_lambda, sparsity_percentage, continuity_lambda):
        #gen_loss = sparsity_loss + continuity_loss + abs(full_text_cls_loss)
        # update gradient
        # gen_loss.requires_grad_(True)
        gen_loss.backward()#！！！！！！
                            
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
        '''
        if args.pred_div==1:
            opt_pred.step()
            opt_pred.zero_grad()

        n1 = 0
        n2 = 0
        n3 = 0
        for idx,p in model.pred_encoder.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.pred_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        for idx,p in model.layernorm2.named_parameters():
            if idx in name3:
                p.requires_grad = True
                n3 += 1
        '''


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
            print('jsd_loss', jsd_loss,"\n sparsity_loss",sparsity_loss,"\n gen_loss",gen_loss)
            print("The val performance: sparsity: %.2f,accuracy: %2f,f1_score :%2f,full_accuracy: %2f" % (100 * annotation_results[0], 100 * annotation_results[1], 100 * annotation_results[2], 100 * annotation_results[3]))
            # print(
            #     "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            #     % (100 * annotation_results[0], 100 * annotation_results[1],

            if annotation_results[3]>best_acc:
                if annotation_results[0] < 0.6:
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



