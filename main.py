import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
from tqdm import tqdm
import logging
from datetime import datetime
import time

from utils import read_data, use_iobes, build_dicts, map_to_idx, batching, PAD_IDX
from model.lstmcrfpa import LstmCrfPa
from model.lstmcrfpasl import LstmCrfPaSl
from model.selector import Selector


def set_seed(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.device.startswith('cuda'):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(
    description='Pytorch implementation of Distant Supervised NER.'
)
parser.add_argument('--device', type=str, default='cuda:3', choices=['cpu', 'cuda:3'])
parser.add_argument('--embedding_file', type=str,
                    default='resource/embedding/pre_trained_100dim.model')
parser.add_argument('--embedding_dim', type=int,
                    default=100)
parser.add_argument('--batch_size', type=int,
                    default=128)
parser.add_argument('--hidden_dim', type=int,
                    default=200)
parser.add_argument('--epochs', type=int,
                    default=800)
parser.add_argument('--learning_rate', type=float,
                    default=0.001)
parser.add_argument('--optimizer', type=str,
                    default='RMSprop')
parser.add_argument('--use_iobes', type=bool,
                    default=True)
parser.add_argument('--from_begin', action='store_true')
parser.add_argument('--dataset', type=str, choices=['ec', 'msra'],
                    default='ec')
parser.add_argument('--max_len', type=int, default=75)
parser.add_argument('--use_pa', action='store_true')
parser.add_argument('--no_distant', action='store_true')
parser.add_argument('--use_selector', action='store_true')


def _logging():
    os.mkdir(logdir)
    logfile = os.path.join(logdir, 'log.log')
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def logging_config(args):
    logger.info('Config:')
    for k, v in vars(args).items():
        logger.info(k + ": " + str(v))
    logger.info("")


def build_embedding_dict(embedding_file, embedding_dim):
    embedding_dict_file = embedding_file + '.pkl'
    if os.path.exists(embedding_dict_file):
        with open(embedding_dict_file, 'rb') as f:
            embedding_dict = pickle.load(f)
        return embedding_dict

    embedding_dict = {}
    with open(embedding_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                continue
            line = line.split()
            word = line[0]
            vec = list(map(float, line[1:]))
            assert len(vec) == embedding_dim
            embedding_dict[word] = np.array(vec).reshape(1, embedding_dim)
    with open(embedding_dict_file, 'wb') as f:
        pickle.dump(embedding_dict, f)
    return embedding_dict


def build_embedding_table(
        embedding_dim,
        embedding_dict,
        dicts
):
    idx_to_char = dicts['idx_to_char']
    char_size = dicts['char_size']

    scale = np.sqrt(3.0 / embedding_dim)
    embedding_table = np.zeros((char_size, embedding_dim))
    count = 0
    for i in range(char_size):
        char = idx_to_char[i]
        if char.lower() in embedding_dict:
            count += 1
            embedding_table[i] = embedding_dict[char.lower()]
        else:
            embedding_table[i] = np.random.uniform(-scale, scale, [1, embedding_dim])
    logger.info('[%d / %d] chars has corresponding pre-trained embeddings.' % (count, char_size))
    logger.info('number of pre-trained embeddings: %d' % len(embedding_dict))
    return embedding_table


def cal_metrics(pred_paths, gold_paths):

    def get_name_entities(path):
        length = len(path)
        name_entitis = set()
        i = 0
        while i < length:
            cur_tag = path[i]
            if cur_tag.startswith('B-'):
                tag = cur_tag[2:]
                in_tag = cur_tag.replace('B', 'I')
                j = i + 1
                while j < length and path[j] == in_tag:
                    j += 1
                name_entitis.add(tag + "-" + str(i) + "-" + str(j))
                i = j
            else:
                i += 1
        return name_entitis

    tp, tp_fp, tp_fn = 0, 0, 0
    for pred_path, gold_path in zip(pred_paths, gold_paths):
        pred_nes = get_name_entities(pred_path)
        gold_nes = get_name_entities(gold_path)
        tp += len(pred_nes.intersection(gold_nes))
        tp_fp += len(pred_nes)
        tp_fn += len(gold_nes)

    return tp, tp_fp, tp_fn


def cal_metrics_iobes(pred_paths, gold_paths):

    def get_name_entities(path):
        name_entitis = set()
        start = -1
        for i in range(len(path)):
            cur_tag = path[i]
            if cur_tag.startswith('B-'):
                start = i
            if cur_tag.startswith('E-'):
                name_entitis.add(str(start) + "-" + str(i) + "-" + cur_tag[2:])
            if cur_tag.startswith('S-'):
                name_entitis.add(str(i) + "-" + str(i) + "-" + cur_tag)
        return name_entitis

    tp, tp_fp, tp_fn = 0, 0, 0
    for pred_path, gold_path in zip(pred_paths, gold_paths):
        pred_nes = get_name_entities(pred_path)
        gold_nes = get_name_entities(gold_path)
        tp += len(pred_nes.intersection(gold_nes))
        tp_fp += len(pred_nes)
        tp_fn += len(gold_nes)

    return tp, tp_fp, tp_fn


def evaluate(tagger, samples, args, dicts):
    tagger.eval()
    idx_to_tag = dicts['idx_to_tag']
    batchs = batching(samples, args)
    with torch.no_grad():
        tp, tp_fp, tp_fn = 0, 0, 0
        for batch in batchs:
            gold_paths = [sample.tag_ids for sample in batch]
            gold_paths = [
                [idx_to_tag[idx] for idx in path]
                for path in gold_paths
            ]
            data = padding(batch, dicts)
            for k, v in data.items():
                data[k] = v.to(args.device)

            pred_paths = tagger(data)
            pred_paths = [
                [idx_to_tag[idx] for idx in path]
                for path in pred_paths
            ]
            if args.use_iobes:
                tp_, tp_fp_, tp_fn_ = cal_metrics_iobes(pred_paths, gold_paths)
            else:
                tp_, tp_fp_, tp_fn_ = cal_metrics(pred_paths, gold_paths)
            tp += tp_
            tp_fp += tp_fp_
            tp_fn += tp_fn_

        precision = 100.0 * tp / tp_fp if tp_fp != 0 else 0
        recall = 100.0 * tp / tp_fn if tp_fn != 0 else 0
        f_score = 2 * precision * recall / (precision + recall) \
            if precision != 0 or recall != 0 else 0
    return precision, recall, f_score


def padding(samples, dicts):
    batch_size = len(samples)
    tag_size = dicts['tag_size']
    pad_hot = [0] * tag_size
    pad_hot[PAD_IDX] = 1
    char_seq_lens = torch.tensor(
        list(map(lambda sap: len(sap.char_ids), samples)),
        dtype=torch.long
    )

    # max_seq_len = torch.max(char_seq_lens).item()
    max_seq_len = args.max_len  # a fixed sentence length, 75 for ec, 100 for msra.

    char_seq_tensor = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.long
    )
    tag_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, tag_size),
        dtype=torch.long
    )

    for i, sample in enumerate(samples):
        cur_len = char_seq_lens[i].item()
        char_seq_tensor[i, 0: cur_len] = torch.tensor(sample.char_ids, dtype=torch.long)
        for j, hot in enumerate(sample.tag_hots):
            tag_seq_tensor[i, j] = torch.tensor(hot, dtype=torch.long)

        for j in range(cur_len, max_seq_len):
            tag_seq_tensor[i, j] = torch.tensor(pad_hot, dtype=torch.long)
    return dict(
        char_seq_tensor=char_seq_tensor,
        tag_seq_tensor=tag_seq_tensor,
        char_seq_lens=char_seq_lens,
    )


def main(args):
    set_seed(args, 0)

    # Prepare data
    train_file = '/'.join(['data', args.dataset, 'train'])
    dev_file = '/'.join(['data', args.dataset, 'dev'])
    test_file = '/'.join(['data', args.dataset, 'test'])
    if args.use_pa:
        train_file_ds = '/'.join(['data', args.dataset, 'ds_pa'])
    else:
        train_file_ds = '/'.join(['data', args.dataset, 'ds_fa'])

    train_data = read_data(train_file)
    dev_data = read_data(dev_file)
    test_data = read_data(test_file)
    train_data_pa = read_data(train_file_ds)

    if args.use_iobes:
        use_iobes(train_data)
        use_iobes(dev_data)
        use_iobes(test_data)
        use_iobes(train_data_pa)

    dicts = build_dicts(
        train_data,
        dev_data,
        test_data,
        train_data_pa,
        args
    )
    idx_to_tag = dicts['idx_to_tag']
    print('tags:', idx_to_tag)

    embedding_dict = build_embedding_dict(
        args.embedding_file,
        args.embedding_dim,
    )
    embedding_table = build_embedding_table(
        args.embedding_dim,
        embedding_dict,
        dicts
    )

    train_samples = map_to_idx(train_data, dicts)
    dev_samples = map_to_idx(dev_data, dicts)
    test_samples = map_to_idx(test_data, dicts)
    if not args.no_distant:
        train_pa_samples = map_to_idx(train_data_pa, dicts)
        train_samples.extend(train_pa_samples)  # in-place operation
        random.shuffle(train_samples)
    print('number of samples:', len(train_samples))

    for i, sample in enumerate(train_samples):
        sample.id = i

    if args.use_selector:
        # Training with selector
        logger.info('Using LstmCrfPaSl.')
        model_file = 'checkpoint-pa-sl-{}.pt'.format(args.dataset)
        tagger = LstmCrfPaSl(args, dicts, embedding_table).to(args.device)
        selector = Selector(args).to(args.device)
        opt = optim.RMSprop(tagger.parameters(), lr=args.learning_rate)
        opt_sl = optim.Adam(selector.parameters(), lr=args.learning_rate)

        max_f_score = 0.0
        if not args.from_begin:
            checkpoint = torch.load(model_file)
            tagger.load_state_dict(checkpoint['tagger_state_dict'])
            opt.load_state_dict(checkpoint['opt_state_dict'])
            selector.load_state_dict(checkpoint['selector_state_dict'])
            opt_sl.load_state_dict(checkpoint['opt_sl_state_dict'])
            max_f_score = checkpoint['max_f_score']

        criterion = nn.BCEWithLogitsLoss()

        batch_size = args.batch_size
        random.shuffle(train_samples)
        select_track_all = dict()
        for epoch in range(1, args.epochs + 1):
            select_track = dict()
            start_time = time.time()
            batchs = []
            batch = []
            action_batch = torch.FloatTensor().to(args.device)
            alpha_batch = torch.FloatTensor().to(args.device)
            select_loss = 0
            # select batchs and update selector
            for idx in range(len(train_samples)):
                sample = train_samples[idx]
                if len(batch) == batch_size:
                    batchs.append(batch)
                    if action_batch.size(0) > 0:
                        selector.train()

                        data = padding(batch, dicts)
                        for k, v in data.items():
                            data[k] = v.to(args.device)

                        with torch.no_grad():
                            nll = tagger.neg_log_likelihood(data, dicts)
                        mean_reward = -nll / batch_size  # reward should not substract all_path_score
                        log_prob = criterion(alpha_batch, action_batch)
                        loss = mean_reward * log_prob
                        select_loss += loss.item()
                        loss.backward()
                        opt_sl.step()
                        selector.zero_grad()

                        batch = []
                        action_batch = torch.FloatTensor().to(args.device)
                        alpha_batch = torch.FloatTensor().to(args.device)

                elif sample.sign == 0:
                    batch.append(sample)
                else:
                    data = padding([sample], dicts)
                    for k, v in data.items():
                        data[k] = v.to(args.device)

                    with torch.no_grad():
                        state_rep = tagger.encode(data)
                    alpha = selector(state_rep).view(1)
                    alpha_batch = torch.cat([alpha_batch, alpha], dim=0)
                    if alpha > 0.5:
                        batch.append(sample)
                        action_batch = torch.cat(
                            [action_batch, torch.tensor([1], dtype=torch.float).to(args.device)], dim=0
                        )
                        if sample.id not in select_track:
                            select_track[sample.id] = 1
                        else:
                            select_track[sample.id] += 1
                    else:
                        action_batch = torch.cat(
                            [action_batch, torch.tensor([0], dtype=torch.float).to(args.device)], dim=0
                        )

            if len(batch) > 0:
                batchs.append(batch)
                if action_batch.size(0) > 0:
                    selector.train()
                    data = padding(batch, dicts)
                    for k, v in data.items():
                        data[k] = v.to(args.device)

                    with torch.no_grad():
                        nll = tagger.neg_log_likelihood(data, dicts)
                    mean_reward = -nll / batch_size
                    log_prob = criterion(alpha_batch, action_batch)
                    loss = mean_reward * log_prob
                    select_loss += loss.item()
                    loss.backward()
                    opt_sl.step()
                    selector.zero_grad()
            print('Select Loss: ', select_loss)
            select_track_all[epoch] = select_track

            # Update tagger
            epoch_loss = 0

            for idx in np.random.permutation(len(batchs)):
                tagger.train()
                tagger.zero_grad()
                data = padding(batchs[idx], dicts)
                for k, v in data.items():
                    data[k] = v.to(args.device)

                nll = tagger.neg_log_likelihood(data, dicts)
                epoch_loss += nll.item()
                nll.backward()
                opt.step()
            end_time = time.time()
            logger.info(
                '[Epoch %d / %d] [Loss: %f] [Time: %f]' %
                (epoch, args.epochs, epoch_loss, end_time - start_time)
            )

            precision, recall, f_score = evaluate(tagger, dev_samples, args, dicts)
            logger.info(
                '[Dev set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score)
            )

            if f_score > max_f_score:
                torch.save({'tagger_state_dict': tagger.state_dict(),
                            'opt_state_dict': opt.state_dict(),
                            'selector_state_dict': selector.state_dict(),
                            'opt_sl_state_dict': opt_sl.state_dict(),
                            'max_f_score': max_f_score
                            }, model_file)
                max_f_score = f_score
                logger.info('Save the best model.')

            precision, recall, f_score = evaluate(tagger, test_samples, args, dicts)
            logger.info(
                '[test set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score))
            logger.info("")

        with open("select_info", 'wb') as f:
            pickle.dump(select_track_all, f)

        # the best model
        with torch.no_grad():
            checkpoint = torch.load(model_file)
            tagger.load_state_dict(checkpoint['tagger_state_dict'])

            logger.info('The best:')
            precision, recall, f_score = evaluate(tagger, dev_samples, args, dicts)
            logger.info(
                '[Dev set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score)
            )
            precision, recall, f_score = evaluate(tagger, test_samples, args, dicts)
            logger.info(
                '[test set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score))
            logger.info("")
    else:
        # Training without selector
        tagger = LstmCrfPa(args, dicts, embedding_table).to(args.device)

        opt = optim.RMSprop(tagger.parameters(), lr=args.learning_rate)

        random.shuffle(train_samples)
        batchs = batching(train_samples, args)
        max_f_score = 0.0
        if args.no_distant:
            model_file = 'checkpoint-no-ds-{}.pt'.format(args.dataset)
        elif args.use_pa:
            model_file = 'checkpoint-pa-{}.pt'.format(args.dataset)
        else:
            model_file = 'checkpoint-{}.pt'.format(args.dataset)

        if not args.from_begin:
            checkpoint = torch.load(model_file)
            tagger.load_state_dict(checkpoint['tagger_state_dict'])
            opt.load_state_dict(checkpoint['opt_state_dict'])
            max_f_score = checkpoint['max_f_score']

        for i in range(1, args.epochs + 1):
            epoch_loss = 0
            start_time = time.time()
            for idx in np.random.permutation(len(batchs)):
                tagger.train()
                tagger.zero_grad()
                data = padding(batchs[idx], dicts)
                for k, v in data.items():
                    data[k] = v.to(args.device)

                nll = tagger.neg_log_likelihood(data, dicts)
                epoch_loss += nll.item()
                nll.backward()
                opt.step()
            end_time = time.time()
            logger.info(
                '[Epoch %d / %d] [Loss: %f] [Time: %f]' %
                (i, args.epochs, epoch_loss, end_time - start_time)
            )

            precision, recall, f_score = evaluate(tagger, dev_samples, args, dicts)
            logger.info(
                '[Dev set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score)
            )

            if f_score > max_f_score:
                torch.save({'tagger_state_dict': tagger.state_dict(),
                            'opt_state_dict': opt.state_dict(),
                            'max_f_score': max_f_score
                            }, model_file)
                max_f_score = f_score
                logger.info('Save the best model.')

            precision, recall, f_score = evaluate(tagger, test_samples, args, dicts)
            logger.info(
                '[test set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score))
            logger.info("")

        # the best model
        with torch.no_grad():
            checkpoint = torch.load(model_file)
            tagger.load_state_dict(checkpoint['tagger_state_dict'])

            logger.info('The best:')
            precision, recall, f_score = evaluate(tagger, dev_samples, args, dicts)
            logger.info(
                '[Dev set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score)
            )
            precision, recall, f_score = evaluate(tagger, test_samples, args, dicts)
            logger.info(
                '[test set] [precision %f] [recall %f] [fscore %f]' %
                (precision, recall, f_score))
            logger.info("")


if __name__ == '__main__':
    args = parser.parse_args()
    global logdir
    logdir = '-'.join([
        'log/log',
        args.dataset,
        datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    ])
    if args.dataset == 'msra':
        args.max_len = 100
    _logging()
    logging_config(args)
    main(args)
