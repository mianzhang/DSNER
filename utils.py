import os
import re
import pickle
import torch
from tqdm import tqdm
from sample import Sample

PAD = '<PAD>'
START = '<START>'
STOP = '<STOP>'
UNK = '<UNK>'
PAD_IDX = 0


def read_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        chars, tags = [], []
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                data.append((chars, tags))
                chars, tags = [], []
                continue
            line = line.split()
            if len(line) != 2:
                continue
            char = re.sub('\d', '0', line[0])
            chars.append(char)
            tag = line[1]
            if tag == 'UNK':
                tag = UNK
            tags.append(tag)

    return data


def use_iobes(data):
    for words, tags in data:
        for i in range(len(tags)):
            cur_tag = tags[i]
            if i == len(tags) - 1:
                if cur_tag.startswith('B-'):
                    tags[i] = cur_tag.replace('B-', 'S-')
                elif cur_tag.startswith('I'):
                    tags[i] = cur_tag.replace('I-', 'E-')
            else:
                next_tag = tags[i + 1]
                if cur_tag.startswith('B-'):
                    if next_tag.startswith('O') or next_tag.startswith('B-') or next_tag == UNK:
                        tags[i] = cur_tag.replace('B-', 'S-')
                elif cur_tag.startswith('I'):
                    if next_tag.startswith('O') or next_tag.startswith('B-') or next_tag == UNK:
                        tags[i] = cur_tag.replace('I-', 'E-')


def build_dicts(train_data, dev_data, test_data, train_data_ds, args):
    if args.no_distant:
        map_dict_file = '/'.join(['resource', 'mapping', args.dataset + '-no-ds' + '.pkl'])
    elif args.use_pa:
        map_dict_file = '/'.join(['resource', 'mapping', args.dataset + '-pa' + '.pkl'])
    else:
        map_dict_file = '/'.join(['resource', 'mapping', args.dataset + '.pkl'])
    if os.path.exists(map_dict_file):
        with open(map_dict_file, 'rb') as f:
            map_dict = pickle.load(f)
            return map_dict
    tag_to_idx = {PAD: 0, UNK: 1}
    idx_to_tag = [PAD, UNK]
    char_to_idx = {PAD: 0}
    idx_to_char = [PAD]

    if args.no_distant:
        data = train_data + dev_data + test_data
    else:
        data = train_data + dev_data + test_data + train_data_ds
    for chars, tags in data:
        for char, tag in zip(chars, tags):
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx)
                idx_to_char.append(char)
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
                idx_to_tag.append(tag)

    tag_to_idx[START] = len(tag_to_idx)
    tag_to_idx[STOP] = len(tag_to_idx)
    idx_to_tag.append(START)
    idx_to_tag.append(STOP)

    map_dict = dict(
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag,
        char_size=len(idx_to_char),
        tag_size=len(idx_to_tag)
    )
    with open(map_dict_file, 'wb') as f:
        pickle.dump(map_dict, f)
    return map_dict


def map_to_idx(data, dicts):
    samples = []
    char_to_idx = dicts['char_to_idx']
    tag_to_idx = dicts['tag_to_idx']
    for chars, tags in data:
        char_ids = [char_to_idx[char] for char in chars]
        tag_ids = [tag_to_idx[tag] for tag in tags]
        samples.append(Sample(char_ids, tag_ids, dicts))
    return samples


def batching(samples, args):
    batch_size = args.batch_size
    batched_samples = []
    if len(samples) % batch_size == 0:
        batch_num = len(samples) // batch_size
    else:
        batch_num = len(samples) // batch_size + 1

    for i in range(batch_num):
        batched_samples.append(
            samples[i * batch_size: (i + 1) * batch_size]
        )

    return batched_samples


def log_sum_exp_pytorch(t):
    max_score, _ = torch.max(t, dim=-1)
    return max_score + torch.log(torch.sum(torch.exp(t - max_score.unsqueeze(-1)), -1))


def log_sum_exp_pa(t, mask):
    mask = torch.eq(mask, 0)
    max_score, _ = torch.max(t, dim=-1)
    exp_t = torch.exp(t - max_score.unsqueeze(-1))
    masked_exp_t = torch.masked_fill(exp_t, mask, 0)
    sum_t = torch.sum(masked_exp_t, dim=-1)
    mask_sum = torch.eq(sum_t, 0)
    masked_sum_t = torch.masked_fill(sum_t, mask_sum, 1)
    log_t = torch.log(masked_sum_t)
    res = log_t + max_score
    masked_res = torch.masked_fill(res, mask_sum, 0)
    return masked_res
