# encoding: utf-8
from typing import List

import codecs
import unicodedata
from typing import List, Optional, Dict
class Tokenizer:
    """
    Abstract base class for all implemented tokenizer.
    """

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token sequence
        Args:
            text: target text sample
        Returns:
            List of tokens in this sample
        """
        return text.split(' ')
TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


class BertTokenizer(Tokenizer):
    """
    Bert Like Tokenizer, ref: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py
    """

    def __init__(self,
                 *,
                 token_dict: Optional[Dict[str, int]] = None,
                 token_cls: str = TOKEN_CLS,
                 token_sep: str = TOKEN_SEP,
                 token_unk: str = TOKEN_UNK,
                 pad_index: int = 0,
                 cased: bool = False) -> None:
        """
        Initialize tokenizer.
        Args:
            token_dict: A dict maps tokens to indices.
            token_cls: The token represents classification.
            token_sep: The token represents separator.
            token_unk: The token represents unknown token.
            pad_index: The index to pad.
            cased: Whether to keep the case.
        """
        self._token_dict: Dict[str, int]

        if token_dict:
            self._token_dict = token_dict
        else:
            self._token_dict = {}

        self._token_dict_inv: Dict[int, str] = {v: k for k, v in self._token_dict.items()}
        self._token_cls: str = token_cls
        self._token_sep: str = token_sep
        self._token_unk: str = token_unk
        self._pad_index: int = pad_index
        self._cased: bool = cased

    @classmethod
    def load_from_vocab_file(cls, vocab_path: str) -> 'BertTokenizer':
        token2idx: Dict[str, int] = {}
        with codecs.open(vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token2idx[token] = len(token2idx)
        return BertTokenizer(token_dict=token2idx)

    def tokenize(self, text: str) -> List[str]:
        """
        Split text to tokens.
        Args:
            text: text to tokenize.
        Returns:
            A list of strings.
        """
        tokens = self._tokenize(text)
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        if len(self._token_dict) > 0:
            tokens = []
            for word in spaced.strip().split():
                tokens += self._word_piece_tokenize(word)
            return tokens
        else:
            return spaced.strip().split()

    def _word_piece_tokenize(self, word: str) -> List[str]:
        if word in self._token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens

    @staticmethod
    def _is_punctuation(ch: str) -> bool:  # noqa: E127
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch: str) -> bool:
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch: str) -> bool:
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch: str) -> bool:
        return unicodedata.category(ch) in ('Cc', 'Cf')