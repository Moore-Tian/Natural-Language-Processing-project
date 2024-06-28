from typing import Any


class word_mapping():
    def __init__(self, dataset):
        self.encode_mapping = {}
        self.num_word = 1

        for words, _ in dataset:
            for word in words:
                if word not in self.encode_mapping:
                    self.encode_mapping[word] = self.num_word
                    self.num_word += 1
        self.encode_mapping["<PAD>"] = self.num_word
        self.encode_mapping["<unk>"] = 0
        self.num_word += 1
        self.decode_mapping = {val: key for key, val in self.encode_mapping.items()}

    def __len__(self):
        return self.num_word

    def encode(self, words):
        return [self.encode_mapping[word] if word in self.encode_mapping else 0 for word in words]
    
    def decode(self, codes):
        return [self.decode_mapping[code] if code in self.decode_mapping else '<unk>' for code in codes]