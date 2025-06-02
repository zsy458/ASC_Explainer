from train import Instructor
import os
import sys
import copy
import logging
import argparse
import torch
import transformers as ppb
from rational_model.model import Bert_grus
from tensorboardX import SummaryWriter

from time import strftime, localtime
from torch.utils.data import DataLoader

import time
from rational_model.train_util_bert_bcr import train_bert_bcr_onegpu_distillation,validate_share_bert_onegpu
from models import TD_LSTM, ATAE_LSTM, BERT_SPC, LCF_BERT, RelationalAttentionBertClassifier, SSEGCNBertClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



def parse():
    model_classes = {
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'bert_spc': BERT_SPC,
        'lcf_bert': LCF_BERT,
        'ssegcn':SSEGCNBertClassifier,
        'aparn': RelationalAttentionBertClassifier
    }
    
    dataset_files = {
        'twitter': {
            'train': './dataset/Tweets_corenlp/train_with_term_num_duiqi.xml.seg',
            'test': './dataset/Tweets_corenlp/test_with_term_num_duiqi.xml.seg'
        },
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train_with_term_num_duiqi.xml.seg',
            'test': './dataset/Restaurants_corenlp/test_with_term_num_duiqi.xml.seg'
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train_with_term_num_duiqi.xml.seg',
            'test': './dataset/Laptops_corenlp/test_with_term_num_duiqi.xml.seg'
        },
        'mams': {
            'train': './dataset/MAMS_corenlp/train_with_term_num_duiqi.xml.seg',
            'test': './dataset/MAMS_corenlp/test_with_term_num_duiqi.xml.seg'
        },
        'syntactic_restaurant': {
            'train': './dataset/Restaurants_corenlp/train_with_term_num.json',
            'test': './dataset/Restaurants_corenlp/test_with_term_num.json',
        },
        'syntactic_laptop': {
            'train': './dataset/Laptops_corenlp/train_with_term_num.json',
            'test': './dataset/Laptops_corenlp/test_with_term_num.json'
        },
        'syntactic_twitter': {
            'train': './dataset/Tweets_corenlp/train_with_term_num.json',
            'test': './dataset/Tweets_corenlp/test_with_term_num.json',
        },
        'syntactic_mams': {
            'train': './dataset/MAMS_corenlp/train_with_term_num.json',
            'test': './dataset/MAMS_corenlp/test_with_term_num.json',
        }
    }
    input_colses = {
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices','attention_mask'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'ssegcn': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask','short_mask'],
        'aparn': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end',
                           'adj_matrix', 'edge_adj', 'src_mask', 'aspect_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    
    # Hyperparameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size [default: 100]')
        # model parameters
    parser.add_argument('--gen_acc',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--pred_div',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--gen_sparse',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--div',
                        type=str,
                        default='kl',
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--freeze_bert',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--sp_norm',
                        type=int,
                        default=0,
                        help='0:rnp,1:norm')
    parser.add_argument('--dis_lr',
                        type=float,
                        default=0,
                        help='0:rnp,1:dis')
    parser.add_argument('--save',
                        type=int,
                        default=1,
                        help='save model, 0:do not save, 1:save')
    # parser.add_argument('--cell_type',
    #                     type=str,
    #                     default="electra",
    #                     help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--rnn_num_layers',  #参数名变了，别忘了
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=300,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')


    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--encoder_lr',
                        type=float,
                        default=0.00001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--fc_lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    # parser.add_argument('--gpu',
    #                     type=int,
    #                     default=1,
    #                     help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')



    parser.add_argument('--model_name', default='APARN', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--attn_dropout', type=float, default=0.3, help='Attention layer dropout rate.')
    parser.add_argument('--device', default='cuda:1', type=str, help='cpu, cuda')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--ex_model_path', type=str, default='')
    opt = parser.parse_args()

    #gen参数
    opt.cell_type = 'bert'

    #cla 参数
    opt.amr_edge_stoi = './APARN/stoi.pt'
    opt.amr_edge_pt = './APARN/embedding.pt'
    opt.amr_edge_dim = 1024
    opt.max_seq_len = 85
    opt.feature_type = '1+A'
    opt.edge = 'normal'
    opt.edge_dropout = opt.bert_dropout
    opt.final_dropout = opt.bert_dropout
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.attention_heads = 5
    opt.bert_dim = 768
    opt.num_layers = 1
    opt.embed_dim = 300
    # opt.amr_edge_dim = 64
    opt.polarities_dim = 3
    opt.parseamr = True
    opt.direct = False
    opt.part = 1
    opt.dim_heads = 64
    opt.valset_ratio = 0
    # opt.model_name = 'relationalbert'
    opt.gcn_dropout = 0.1
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.local_context_focus = 'cdm'
    opt.SRD =3
    # opt.initializer = initializers[opt.initializer]
    # opt.optimizer = optimizers[opt.optimizer]
    return opt



def main():
    opt = parse()

    print("choice cuda:{}".format(opt.cuda))
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    # opt.device = torch.device('cpu')

    # set random seed
    # setup_seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)


    if not os.path.exists('./absa/log'):
        os.makedirs('./absa/log', mode=0o777)
    log_file = '{}-{}-{}.log0'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./absa/log', log_file)))

    class_old = Instructor(opt)
    classifier = class_old.model
    classifier.load_state_dict(torch.load(opt.model_path,map_location=opt.device))
    classifier.eval()
    classifier.to(opt.device)
    for name,p in classifier.named_parameters():
        p.requires_grad=False#冻结参数

    tokenizer_class, pretrained_weights = (ppb.BertTokenizerFast, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    opt.hidden_dim=768
    train_loader = class_old.trainset
    train_loader = DataLoader(dataset=class_old.trainset, batch_size=opt.batch_size, shuffle=False)
    dev_loader = DataLoader(dataset=class_old.testset, batch_size=opt.batch_size, shuffle=False)

    annotation_loader = dev_loader
    writer=SummaryWriter(opt.writer)
    model=Bert_grus(opt)
    model.to(opt.device)

    lr_gen_encoder=opt.encoder_lr
    lr_gen_fc=opt.fc_lr
    if opt.dis_lr == 2:
        lr_pred_encoder=opt.encoder_lr/2
        lr_pred_fc=opt.fc_lr/2
    elif opt.dis_lr == 3:
        lr_pred_encoder = opt.encoder_lr / 3
        lr_pred_fc = opt.fc_lr / 3
    else:
        lr_pred_encoder = opt.encoder_lr
        lr_pred_fc = opt.fc_lr

    para_gen_encoder=filter(lambda p: p.requires_grad, model.gen_encoder.parameters())
    para_gen_fc=filter(lambda p: p.requires_grad, model.gen.parameters())
    para_gen_linear=filter(lambda p: p.requires_grad, model.gen_linear.parameters())

    para_gen=[{'params':para_gen_encoder,'lr': lr_gen_encoder},
        {'params':para_gen_fc,'lr': lr_gen_fc},
        {'params':para_gen_linear,'lr': lr_gen_fc}]
    
    opt_gen=torch.optim.Adam(para_gen)

    strat_time=time.time()
    best_all = 0
    f1_best_dev = [0]
    best_dev_epoch = [0]
    acc_best_dev = [0]
    grad=[]
    grad_loss=[]
    best_model = copy.deepcopy(model)
    best_acc = 0
    # model.load_state_dict(torch.load(opt.ex_model_path,map_location=opt.device)) 
    # val_sparsity, val_accuracy, val_f1_score, val_accuracy_to_full_model = validate_share_bert_onegpu(model, classifier, annotation_loader, opt.device, opt, False)
    # a = hh
    for epoch in range(opt.epochs):

        start = time.time()
        f1_score, accuracy, accuracy_model, best_model, best_acc = train_bert_bcr_onegpu_distillation(model,classifier, opt_gen, train_loader, opt.device, opt, annotation_loader,best_acc,best_model)
        end = time.time()

        logger.info('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
        # print('gen_lr={}, pred_lr={}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[3]['lr']))
        logger.info("traning epoch:{} f1-score:{:.4f} accuracy:{:.4f} accuracy_model:{:.4f}".format(epoch, f1_score, accuracy, accuracy_model))
        writer.add_scalar('train_acc',accuracy,accuracy_model,epoch)
        writer.add_scalar('time',time.time()-strat_time,epoch)
        model.eval()

        # precision, recall, f1_score, accuracy=dev_bert_bcr_onetigpu(model,classifier,  dev_loader, opt.device, opt,(writer,epoch))#检验分类

        # logger.info("Validate")
        # logger.info("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall, precision, f1_score, accuracy))

        # writer.add_scalar('dev_acc',accuracy,epoch)
        # print("Validate Sentence")
        # validate_dev_sentence(model, dev_loader, device,(writer,epoch))

        val_sparsity, val_accuracy, val_f1_score, val_accuracy_to_full_model = validate_share_bert_onegpu(model, classifier, annotation_loader, opt.device, opt, False)

        logger.info("Validate")
        logger.info("The val performance: sparsity: %.4f,accuracy:%4f,f1_score:%4f, accuracy_to_model: %4f" % (100 * val_sparsity, 100 * val_accuracy, 100 * val_f1_score, 100 * val_accuracy_to_full_model))

        # logger.info(
            # "The annotation performance: sparsity: %.4f"
            # % (100 * annotation_results))
        # writer.add_scalar('f1',100 * annotation_results[3],epoch)
        writer.add_scalar('sparsity', val_sparsity,epoch)
        writer.add_scalar('accuracy', val_accuracy,epoch)
        # logger.info("Precision, Recall and F1-Score...")
        # logger.info(test_report)
        # logger.info("Confusion Matrix...")
        # logger.info(test_confusion)
        # writer.add_scalar('p', 100 * annotation_results[1], epoch)
        # writer.add_scalar('r', 100 * annotation_results[2], epoch)

        # print("Annotation Sentence")
        # validate_annotation_sentence(model, annotation_loader, device)
        # print("Rationale")
        # validate_rationales(model, annotation_loader, device,(writer,epoch))
        if val_accuracy>acc_best_dev[-1]:
            acc_best_dev.append(val_accuracy)
            best_dev_epoch.append(epoch)

        if best_all<val_accuracy:
            best_all=val_accuracy
        if val_accuracy >best_acc: 
            if val_sparsity < 0.5:
                best_acc = val_accuracy
                best_model = copy.deepcopy(model)


    logger.info('[train_end]')
    logger.info(best_all)
    logger.info(acc_best_dev)
    logger.info(best_dev_epoch)
    logger.info(f1_best_dev)

    save_path = './trained_model/{}_{}_sp{}_con{}_acc{:.4f}.pkl'.format(opt.model_name,opt.dataset,opt.sparsity_lambda,opt.continuity_lambda,best_acc)
    torch.save(best_model.state_dict(),save_path)
    logger.info('save_path:')
    logger.info(save_path)
    logger.info('save the model')

    model = best_model
    model.eval()
    test_report,test_confusion,sparsity, accuracy,f1_score, accuracy_to_full_model, f1_score_to_full_model = validate_share_bert_onegpu(best_model, classifier, annotation_loader, opt.device, opt, True)
    logger.info('train end!')
    logger.info('best_acc:%.4f'%(best_acc))
    logger.info("test")
    logger.info(
        "The test performance: sparsity: %.2f,accuracy: %2f,f1_score: %2f, accuracy_to_full_model: %2f, f1_score_to_full_model: %2f"
        % (100 * sparsity, 100 * accuracy, 100 * f1_score, 100 * accuracy_to_full_model, 100 * f1_score_to_full_model))
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)

if __name__ == '__main__':
    main()
