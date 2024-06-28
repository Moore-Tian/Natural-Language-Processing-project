from typing import Any


class tag_mapping():
    def __init__(self):
        self.encode_mapping = {
            'O': 0,
            'S-GPE': 1,
            'S-PER': 2,
            'B-ORG': 3,
            'E-ORG': 4,
            'S-ORG': 5,
            'M-ORG': 6,
            'S-LOC': 7,
            'E-GPE': 8,
            'B-GPE': 9,
            'B-LOC': 10,
            'E-LOC': 11,
            'M-LOC': 12,
            'M-GPE': 13,
            'B-PER': 14,
            'E-PER': 15,
            'M-PER': 16,
            '<PAD>': 17
        }
        self.num_tag = len(self.encode_mapping)
        self.decode_mapping = {value: key for key, value in self.encode_mapping.items()}

    def encode(self, tags):
        return [self.encode_mapping[tag] for tag in tags]

    def decode(self, codes):
        return [self.decode_mapping[code] for code in codes]

    def __len__(self):
        return self.num_tag