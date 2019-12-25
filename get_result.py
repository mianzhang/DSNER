# -*- coding: utf-8 -*-

import os
import re

dirs = os.listdir("log")

class Epochres:

    def __init__(
            self,
            dev_precision,
            dev_recall,
            dev_fscore,
            test_precision,
            test_recall,
            test_fscore
    ):
        self.dev_precision = dev_precision
        self.dev_recall = dev_recall
        self.dev_fscore = dev_fscore
        self.test_precision = test_precision
        self.test_recall = test_recall
        self.test_fscore = test_fscore

out_dir = "best_results"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for dir in dirs:
    res_all = []
    fpath = '/'.join(["log", dir, "log.log"])
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[13: 16] == "Dev":
                pattern = "\d+\.?\d*"
                dev_res = list(map(float, re.findall(pattern, line)))
                test_res = list(map(float, re.findall(pattern, lines[i + 1])))
                res_all.append(Epochres(*dev_res, *test_res))
    res_all = reversed(sorted(res_all, key=lambda s: s.dev_fscore))
    outf = '/'.join([out_dir, dir + ".best"])
    with open(outf, 'w') as f:
        for i, best_res in enumerate(res_all):
            f.write("rank: {}".format(i))
            f.write('\n')
            f.write("dev_precision: {}".format(best_res.dev_precision))
            f.write('\t')
            f.write("dev_recall: {}".format(best_res.dev_recall))
            f.write('\t')
            f.write("dev_fscore: {}".format(best_res.dev_fscore))
            f.write('\n')
            f.write("test_precision: {}".format(best_res.test_precision))
            f.write('\t')
            f.write("test_recall: {}".format(best_res.test_recall))
            f.write('\t')
            f.write("test_fscore: {}".format(best_res.test_fscore))
            f.write('\n\n')
