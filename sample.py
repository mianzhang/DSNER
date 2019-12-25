PAD = '<PAD>'
START = '<START>'
STOP = '<STOP>'
UNK = '<UNK>'
PAD_IDX = 0


class Sample:

    def __init__(self, char_ids, tag_ids, dicts):
        self.id = -1
        self.char_ids = char_ids
        self.tag_ids = tag_ids
        self.sign = 0
        self.tag_size = dicts['tag_size']
        self.tag_to_idx = dicts['tag_to_idx']
        self.char_to_idx = dicts['char_to_idx']
        self.UNK_IDX = self.tag_to_idx[UNK]
        self.PAD_IDX = self.tag_to_idx[PAD]
        self.START_IDX = self.tag_to_idx[START]
        self.STOP_IDX = self.tag_to_idx[STOP]
        self.tag_hots = self._get_tag_hots(tag_ids)

    def _get_tag_hots(self, tag_ids):
        tag_hots = []
        for idx in tag_ids:
            if idx != self.UNK_IDX:
                hot = [0] * self.tag_size
                hot[idx] = 1
            else:
                self.sign = 1
                hot = [1] * self.tag_size
                hot[self.UNK_IDX] = 0
                hot[self.PAD_IDX] = 0
                hot[self.START_IDX] = 0
                hot[self.STOP_IDX] = 0
            tag_hots.append(hot)
        return tag_hots
