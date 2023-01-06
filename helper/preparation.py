from model import *


def prepare_data(src, target):
    nick_id, pos_id, neg_id, rating = src
    hist_item, hist_mask, neg_items, hist_rate = target
    return nick_id, pos_id, neg_id, rating, hist_item, hist_mask, neg_items, hist_rate


def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


def get_model(dataset, model_type, user_count, item_count, emb_dim, hid_dim, batch_size, maxlen):
    if model_type == 'DNN':
        model = Model_DNN(user_count, item_count, emb_dim, hid_dim, batch_size, maxlen)
    elif model_type == 'MF':
        model = Model_MF(user_count, item_count, emb_dim, hid_dim, batch_size, maxlen)
    elif model_type == 'SVD':
        model = Model_SVD(user_count, item_count, emb_dim, hid_dim, batch_size, maxlen)
    elif model_type == 'GRU4REC':
        model = Model_GRU4REC(user_count, item_count, emb_dim, hid_dim, batch_size, maxlen)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(user_count, item_count, emb_dim, hid_dim, batch_size, args.num_interest,
                           maxlen,
                           relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(user_count, item_count, emb_dim, hid_dim, batch_size,
                                 args.num_interest,
                                 maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(user_count, item_count, emb_dim, hid_dim, batch_size,
                                 args.num_interest,
                                 maxlen)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model


def get_exp_name(dataset, model_type, batch_size, lr, emb_dim, maxlen, perc, save=True):
    # extr_name = input('Please input the experiment name: ')
    extr_name = 'haolun'
    para_name = '_'.join([dataset, model_type, 'd' + str(emb_dim), 'len' + str(maxlen), 'p' + str(perc)])
    exp_name = para_name + '_' + extr_name

    # while os.path.exists('runs/' + exp_name) and save:
    #     # flag = input('The exp name already exists. Do you want to cover? (y/n)')
    #     flag = input('The exp name already exists. Do you want to cover? (y/n)')
    #     if flag == 'y' or flag == 'Y':
    #         shutil.rmtree('runs/' + exp_name)
    #         break
    #     else:
    #         extr_name = input('Please input the experiment name: ')
    #         exp_name = para_name + '_' + extr_name

    return exp_name
