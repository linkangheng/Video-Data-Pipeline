# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""
import ipdb

from abc import ABC
from abc import abstractmethod
from typing import List


def vocab_size_with_padding(orig_vocab_size, exp):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = exp.make_vocab_size_divisible_by * exp.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    print(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
        at='RANK_0'
    )
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )


class SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        name = "SentencePieceTokenizer"
        super().__init__(name)

        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _add_special_token(self, t):
        if t not in self._vocab:
            next_id = len(self._vocab)
            self._vocab[t] = next_id
            self._inv_vocab[next_id] = t
        self._special_tokens[t] = self._vocab[t]
        self._inv_special_tokens[self._vocab[t]] = t

    def _initalize(self, vocab_extra_ids):
        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for i in range(len(self._tokenizer)):
            t = self._tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

        # _add_special_token("<CLS>")
        # self._cls_id = self._vocab["<CLS>"]
        # _add_special_token("<SEP>")
        # self._sep_id = self._vocab["<SEP>"]
        # _add_special_token("<EOD>")
        # self._eod_id = self._vocab["<EOD>"]
        # _add_special_token("<MASK>")
        # self._mask_id = self._vocab["<MASK>"]

        # pad_id = self._tokenizer.pad_id()
        # try:
        #     pad_token = self._tokenizer.id_to_piece(pad_id)
        # except IndexError:
        #     pad_token = "<PAD>"
        # _add_special_token(pad_token)
        # self._pad_id = self._vocab[pad_token]

        bos_id = self._tokenizer.bos_id()
        try:
            bos_token = self._tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = "<BOS>"
        self._add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self._tokenizer.eos_id()
        try:
            eos_token = self._tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = "<EOS>"
        self._add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        unk_id = self._tokenizer.unk_id()
        try:
            unk_token = self._tokenizer.id_to_piece(unk_id)
        except IndexError:
            unk_token = "<UNK>"
        self._add_special_token(unk_token)
        self._unk_id = self._vocab[unk_token]

        i_bos_token = "<im_start>"
        i_eos_token = "<im_end>"
        img_patch_token = "<im_patch>"

        self._add_special_token(img_patch_token)
        self._img_patch_id = self._vocab[img_patch_token]

        self._add_special_token(i_bos_token)
        self._img_bos_id = self._vocab[i_bos_token]

        self._add_special_token(i_eos_token)
        self._img_eos_id = self._vocab[i_eos_token]

        dream_token = "<dream>"
        dream_start_token = "<dream_start>"
        dream_end_token = "<dream_end>"

        self._add_special_token(dream_token)
        self._dream_id = self._vocab[dream_token]

        self._add_special_token(dream_start_token)
        self._dream_bos_id = self._vocab[dream_start_token]

        self._add_special_token(dream_end_token)
        self._dream_eos_id = self._vocab[dream_end_token]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    @property
    def cls(self):
        return self._unk_id

    @property
    def sep(self):
        return self._unk_id

    @property
    def pad(self):
        return self._unk_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eos_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._unk_id
    
    @property
    def img_start_token(self):
        return self._img_bos_id

    @property    
    def img_end_token(self):
        return self._img_eos_id

    @property    
    def img_patch_token(self):
        return self._img_patch_id

    @property
    def dream_token(self):
        return self._dream_id
    
    @property
    def dream_start_token(self):
        return self._dream_bos_id
    
    @property
    def dream_end_token(self):
        return self._dream_eos_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]


class StepmmTokenizer(AbstractTokenizer):
    """Step Chat Tokenizer"""

    def __init__(
        self, model_file, name="StepmmTokenizer",
        img_patch_token = "<im_patch>",
        i_bos_token = "<im_start>",
        i_eos_token = "<im_end>",
        dream_token = "<dream>",
        dream_start_token = "<dream_start>",
        dream_end_token = "<dream_end>",
    ):
        super().__init__(name)

        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for idx in range(self._tokenizer.get_piece_size()):
            text = self._tokenizer.id_to_piece(idx)
            self._inv_vocab[idx] = text
            self._vocab[text] = idx

            if self._tokenizer.is_control(idx) or self._tokenizer.is_unknown(idx):
                self._special_tokens[text] = idx
                self._inv_special_tokens[idx] = text

        self._unk_id = self._tokenizer.unk_id()
        self._bos_id = self._tokenizer.bos_id()
        self._eos_id = self._tokenizer.eos_id()

        for token in [
            img_patch_token, i_bos_token, i_eos_token, dream_token, dream_start_token, dream_end_token
        ]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"
            assert token in self._special_tokens, f"Token '{token}' is not a special token"

        self._img_patch_id = self._tokenizer.piece_to_id(img_patch_token)
        self._img_bos_id = self._tokenizer.piece_to_id(i_bos_token)
        self._img_eos_id = self._tokenizer.piece_to_id(i_eos_token)

        self._dream_id = self._tokenizer.piece_to_id(dream_token)
        self._dream_bos_id = self._tokenizer.piece_to_id(dream_start_token)
        self._dream_eos_id = self._tokenizer.piece_to_id(dream_end_token)


    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    # def tokenize(self, text: str) -> List[int]:
    #     return self._tokenizer.encode_as_ids(text)
    
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    def is_special_token(self, idx: int) -> bool:
        return idx in self._inv_special_tokens

    def detokenize_special(self, idx: int) -> str:
        if idx in self._inv_special_tokens:
            return self._inv_special_tokens[idx]
        return ''

    @property
    def pad(self):
        return self._unk_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def img_start_token(self):
        return self._img_bos_id

    @property    
    def img_end_token(self):
        return self._img_eos_id

    @property    
    def img_patch_token(self):
        return self._img_patch_id

    @property
    def dream_token(self):
        return self._dream_id
    
    @property
    def dream_start_token(self):
        return self._dream_bos_id
    
    @property
    def dream_end_token(self):
        return self._dream_eos_id


class StepChatTokenizer(AbstractTokenizer):
    """Step Chat Tokenizer"""

    def __init__(
        self, model_file, name="StepChatTokenizer",
        bot_token="<|BOT|>",  # Begin of Turn
        eot_token="<|EOT|>",  # End of Turn
        call_start_token="<|CALL_START|>",      # Call Start
        call_end_token="<|CALL_END|>",          # Call End
        think_start_token="<|THINK_START|>",    # Think Start
        think_end_token="<|THINK_END|>",        # Think End
        img_start_token="<|IMG_START|>",        # Image Start
        img_end_token="<|IMG_END|>",            # Image End
    ):
        super().__init__(name)

        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for idx in range(self._tokenizer.get_piece_size()):
            text = self._tokenizer.id_to_piece(idx)
            self._inv_vocab[idx] = text
            self._vocab[text] = idx

            if self._tokenizer.is_control(idx) or self._tokenizer.is_unknown(idx):
                self._special_tokens[text] = idx
                self._inv_special_tokens[idx] = text

        self._unk_id = self._tokenizer.unk_id()
        self._bos_id = self._tokenizer.bos_id()
        self._eos_id = self._tokenizer.eos_id()

        for token in [
            bot_token, eot_token, call_start_token, call_end_token,
            think_start_token, think_end_token, img_start_token, img_end_token
        ]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"
            assert token in self._special_tokens, f"Token '{token}' is not a special token"

        self._bot_id = self._tokenizer.piece_to_id(bot_token)
        self._eot_id = self._tokenizer.piece_to_id(eot_token)
        self._call_start_id = self._tokenizer.piece_to_id(call_start_token)
        self._call_end_id = self._tokenizer.piece_to_id(call_end_token)
        self._think_start_id = self._tokenizer.piece_to_id(think_start_token)
        self._think_end_id = self._tokenizer.piece_to_id(think_end_token)
        self._img_start_id = self._tokenizer.piece_to_id(img_start_token)
        self._img_end_id = self._tokenizer.piece_to_id(img_end_token)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode_as_ids(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode_ids(token_ids)

    def is_special_token(self, idx: int) -> bool:
        return idx in self._inv_special_tokens

    def detokenize_special(self, idx: int) -> str:
        if idx in self._inv_special_tokens:
            return self._inv_special_tokens[idx]
        return ''

    @property
    def pad(self):
        return self._unk_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def bot(self):
        return self._bot_id

    @property
    def eot(self):
        return self._eot_id

    @property
    def call_start(self):
        return self._call_start_id

    @property
    def call_end(self):
        return self._call_end_id

    @property
    def think_start(self):
        return self._think_start_id

    @property
    def think_end(self):
        return self._think_end_id

    @property
    def img_start(self):
        return self._img_start_id

    @property
    def img_end(self):
        return self._img_end_id


class Llama2mmTokenizer(AbstractTokenizer):
    """Step Chat Tokenizer"""

    def __init__(
        self, model_file, name="Llama2mmTokenizer",
        img_patch_token = "<im_patch>",
        i_bos_token = "<im_start>",
        i_eos_token = "<im_end>",
    ):
        super().__init__(name)

        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for idx in range(self._tokenizer.get_piece_size()):
            text = self._tokenizer.id_to_piece(idx)
            self._inv_vocab[idx] = text
            self._vocab[text] = idx

            if self._tokenizer.is_control(idx) or self._tokenizer.is_unknown(idx):
                self._special_tokens[text] = idx
                self._inv_special_tokens[idx] = text

        self._unk_id = self._tokenizer.unk_id()
        self._bos_id = self._tokenizer.bos_id()
        self._eos_id = self._tokenizer.eos_id()

        for token in [
            img_patch_token, i_bos_token, i_eos_token
        ]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"
            assert token in self._special_tokens, f"Token '{token}' is not a special token"

        self._img_patch_id = self._tokenizer.piece_to_id(img_patch_token)
        self._img_bos_id = self._tokenizer.piece_to_id(i_bos_token)
        self._img_eos_id = self._tokenizer.piece_to_id(i_eos_token)


    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    # def tokenize(self, text: str) -> List[int]:
    #     return self._tokenizer.encode_as_ids(text)
    
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    def is_special_token(self, idx: int) -> bool:
        return idx in self._inv_special_tokens

    def detokenize_special(self, idx: int) -> str:
        if idx in self._inv_special_tokens:
            return self._inv_special_tokens[idx]
        return ''

    @property
    def pad(self):
        return self._unk_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def img_start_token(self):
        return self._img_bos_id

    @property    
    def img_end_token(self):
        return self._img_eos_id

    @property    
    def img_patch_token(self):
        return self._img_patch_id

if __name__ == '__main__':
    
    tokenizer_model = "/mnt/shared-storage/tenant/open-source/Llama-2-7b-hf/multimodal_tokenizer.model"
    tokenizer = Llama2mmTokenizer(tokenizer_model)
    
    ipdb.set_trace()

    txt = 'This is just another example of how public schools are using a globalist approach to indoctrinate our children. Watering down the greatness of America by blurring our border lines that so many brave men fought to preserve is repulsive. Fighting back against this mentality in our PUBLIC schools should be a priority for every parent and grandparent in America!\n\nIt is widely accepted that American public schools are controlled by liberals. It seems like every day, we see new examples of American schoolchildren being indoctrinated with left-wing ideas.\n\nThis latest example was brought to our attention by a concerned parent.\n\nKindergarten students from PS75, a public school in New York City, recently took part in a class project in which the children were made to create an American flag with the flags of other 22 other nations superimposed over the stripes. Below the flag read the words “We pledge allegiance to an International Flag.”\n\nCheck out the flag the kindergarten class created:<image>This is the type of globalist indoctrination we have come to expect from the public school system, but telling impressionably young American children that their loyalty should lie with some nebulous idea of a global community rather than their own nation is a new low.'
    tokenizer.tokenize(txt)
    



