import csv

class NER_dataset():
    def __init__(self, datapath):
        super().__init__()
        self.sentences = []
        self.word_mapping = None
        self.tag_mapping = None

        current_sentence_words = []
        current_sentence_tags = []
        with open(datapath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                word = row['word']
                sentence_boundary = word.endswith('。') or word.endswith('？') or word.endswith('！')

                current_sentence_words.append(word)
                current_sentence_tags.append(row['expected'])

                if sentence_boundary:
                    self.sentences.append((current_sentence_words, current_sentence_tags))
                    current_sentence_words = []
                    current_sentence_tags = []

    def get_tag_mapping(self, tag_mapping):
        self.tag_mapping = tag_mapping

    def get_word_mapping(self, word_record):
        self.word_mapping = word_record

    def __getitem__(self, idx):
        sentence_words, sentence_tags = self.sentences[idx]
        if self.word_mapping is not None:
            sentence_words = [self.word_mapping[word] if word in self.word_mapping else 0 for word in sentence_words]
        if self.tag_mapping is not None:
            sentence_tags = [self.tag_mapping[tag] for tag in sentence_tags]
        return sentence_words, sentence_tags

    def __len__(self):
        return len(self.sentences)