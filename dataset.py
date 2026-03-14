import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.src_sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.src_eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.src_pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

        self.target_sos_token = torch.tensor([tokenizer_target.token_to_id("[SOS]")], dtype=torch.int64)
        self.target_eos_token = torch.tensor([tokenizer_target.token_to_id("[EOS]")], dtype=torch.int64)
        self.target_pad_token = torch.tensor([tokenizer_target.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> dict:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.src_sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.src_eos_token,
                torch.tensor([self.src_pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add SOS and EOS to the decoder text
        decoder_input = torch.cat(
            [
                self.target_sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.target_pad_token] * dec_num_padding_tokens)
            ],
            dim=0
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.target_eos_token,
                torch.tensor([self.target_pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input, # Seq_len
            'decoder_input': decoder_input, # Seq_len
            'encoder_mask': (encoder_input != self.src_pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.target_pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len, seq_len)
            'label': label, # Seq_len
            'src_text': src_text,
            'target_text': target_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0