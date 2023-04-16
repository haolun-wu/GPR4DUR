# @title preparation

"""Get data and model."""
from model.model_comirec import Model_ComiRec_DR
from model.model_comirec_sa import Model_ComiRec_SA
from model.model_dnn import Model_DNN
from model.model_gru4rec import Model_GRU4REC
from model.model_mf import Model_MF
from model.model_mind import Model_MIND


def prepare_data(src, target):
    nick_id, pos_id, rating, cate_list, cate_rating = src
    hist_item, hist_mask, hist_rate = target
    return (
        nick_id,
        pos_id,
        rating,
        cate_list,
        cate_rating,
        hist_item,
        hist_mask,
        hist_rate,
    )


def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


def get_model(args):
    if args.model_type == 'MF':
        model = Model_MF(args)
    elif args.model_type == 'DNN':
        model = Model_DNN(args)
    elif args.model_type == 'GRU4REC':
        model = Model_GRU4REC(args)
    elif args.model_type == 'MIND':
        model = Model_MIND(args)
    elif args.model_type == 'ComiRec_DR':
        model = Model_ComiRec_DR(args)
    elif args.model_type == 'ComiRec_SA':
        model = Model_ComiRec_SA(args)
    else:
        print('Invalid model_type : %s', args.model_type)
        return
    return model


def get_exp_name(args):
    para_name = (
            args.dataset
            + '/'
            + '_'.join([
        args.model_type,
        'd' + str(args.embedding_dim),
        'len' + str(args.maxlen),
        'lr' + str(args.lr),
        # 'wd' + str(args.wd),
    ])
    )
    exp_name = para_name
    return exp_name
