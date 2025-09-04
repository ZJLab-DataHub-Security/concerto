import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from .types import CPTMessage, SFTMessage

@dataclass
class DataCollatorForCPTRawText:

    tokenizer: MegatronTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> CPTMessage:
        tokens = []
        channels = []
        for feature in features:
            text = feature['text']
            sentence_ids = self.tokenizer.tokenizer(text, add_special_tokens=False)['input_ids']
            if not sentence_ids:
                print(f"tokenizer error sentence_ids is empty :\n {text} \n")
                continue
            if max(sentence_ids) >= self.tokenizer.vocab_size:
                print(f"tokenizer error max(sentence_ids) >= Encoder.tokenizer.vocab_size :\n {text}\n {max(sentence_ids)}")
                continue
            sentence_ids.append(self.tokenizer.eod)
            tokens.append(sentence_ids)
            channel = feature['channel'] if 'channel' in feature else ''
            channels.append(channel)

        return CPTMessage(
            tokens = tokens,
            channels = channels
        )


@dataclass
class DataCollatorForSFTRawText:

    tokenizer: MegatronTokenizer
    max_padding_length: int = 32768

    def __call__(self, features: List[Dict[str, Any]]) -> SFTMessage:
        input_ids = []
        labels = []
        channels = []
        for feature in features:
            text = feature['messages']
            source = self.tokenizer.apply_chat_template(text[:-1])
            full = self.tokenizer.apply_chat_template(text)
            if len(full) >= self.max_padding_length:
                continue
            for t1, t2 in zip(source, full):
                assert t1 == t2, "The user input_ids are not a prefix of the full input_ids!"

            label = [self.tokenizer.pad_token_id] * (len(source)-1) + full[len(source):] + [self.tokenizer.pad_token_id]
            full[-1] = -1 - full[-1]

            input_ids.append(full)
            labels.append(label)
            channel = feature['channel'] if 'channel' in feature else ''
            channels.append(channel)

        return SFTMessage(
            input_ids = input_ids,
            labels = labels,
            channels = channels
        )
