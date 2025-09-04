from dataclasses import dataclass
from typing import List, Tuple, Deque, Dict, Optional
from collections import deque
from .types import CPTMessage, SFTMessage, CPTConcatedMessage, SFTConcatedMessage

@dataclass
class CPTConcatFn:
    batch_size: int
    max_seqlen: int
    pad_token_id: int
    concated: Optional[Dict[str, Deque]] = None

    def __call__(self, data: Optional[CPTMessage]) -> Optional[CPTConcatedMessage]:
        if self.concated is None:
            self.concated = {
                'tokens': deque([[]]),
                'channels': deque([[]]),
                'index': deque([[0, 0]])
            }
        if data is not None:
            batch_tokens = data.tokens
            batch_channels = data.channels
            for b in range(len(batch_tokens)):
                tokens = batch_tokens[b]
                channel = batch_channels[b]
                # concated to max_seqlen+1 because input_ids and labels are obtained by tokens[:-1] and tokens[1:]
                while len(self.concated['tokens'][-1]) + len(tokens) > self.max_seqlen + 1:
                    cutted_len = self.max_seqlen + 1 - len(self.concated['tokens'][-1])
                    last_index = self.concated['index'][-1][1]
                    self.concated['index'].append([last_index, last_index])
                    if cutted_len == 0:
                        self.concated['tokens'].append([])
                        self.concated['channels'].append([])
                        continue
                    self.concated['tokens'][-1].extend(tokens[:cutted_len])
                    self.concated['channels'][-1].append((channel, cutted_len-1))
                    tokens = tokens[cutted_len-1:]
                    self.concated['tokens'].append([])
                    self.concated['channels'].append([])
                self.concated['tokens'][-1].extend(tokens)
                self.concated['channels'][-1].append((channel, len(tokens)))
                self.concated['index'][-1][1] += 1
        if len(self.concated['tokens']) > self.batch_size:
            concated_message = CPTConcatedMessage(result=[], channels=[], length=0)
            begin_index, _ = self.concated['index'][0]
            for _ in range(self.batch_size):
                concated_message.result.append(self.concated['tokens'].popleft())
                concated_message.channels.append(self.concated['channels'].popleft())
                _, end_index = self.concated['index'].popleft()
            concated_message.length = end_index - begin_index
            return concated_message
        return None

@dataclass
class SFTConcatFn:
    batch_size: int
    max_seqlen: int
    pad_token_id: int
    concated: Optional[Dict[str, Deque]] = None

    def __call__(self, data: Optional[SFTMessage]) -> Optional[SFTConcatedMessage]:
        if self.concated is None:
            self.concated = {
                'input_ids': deque([[]]),
                'labels': deque([[]]),
                'channels': deque([[]]),
                'index': deque([[0, 0]])
            }
        if data is not None:
            assert len(data.input_ids) == len(data.labels), f"batchsize not match, input_ids: {len(data.input_ids)} - labels: {len(data.labels)}"
            for b in range(len(data.input_ids)):
                input_ids = data.input_ids[b]
                labels = data.labels[b]
                channel = data.channels[b]
                assert len(input_ids) == len(labels), f"seqlen not match, input_ids: {len(input_ids)} - labels: {len(labels)}"
                if len(self.concated['input_ids'][-1]) + len(input_ids) > self.max_seqlen:
                    if len(input_ids) > self.max_seqlen:
                        continue
                    concated_len = len(self.concated['input_ids'][-1])
                    self.concated['input_ids'][-1].extend([self.pad_token_id] * (self.max_seqlen - concated_len))
                    self.concated['labels'][-1].extend([-100] * (self.max_seqlen - concated_len))
                    self.concated['input_ids'].append(input_ids)
                    self.concated['labels'].append(labels)
                    self.concated['channels'].append([(channel, len(input_ids))])
                    last_index = self.concated['index'][-1][1]
                    self.concated['index'].append([last_index+1, last_index+1])
                else:
                    self.concated['input_ids'][-1].extend(input_ids)
                    self.concated['labels'][-1].extend(labels)
                    self.concated['channels'][-1].append((channel, len(input_ids)))
                    self.concated['index'][-1][1] += 1
        if len(self.concated['input_ids']) > self.batch_size:
            concated_message = SFTConcatedMessage(result=[], channels=[], length=0)
            begin_index, _ = self.concated['index'][0]
            for _ in range(self.batch_size):
                input_ids = self.concated['input_ids'].popleft()
                labels = self.concated['labels'].popleft()
                channels = self.concated['channels'].popleft()
                concated_message.result.append(input_ids+labels)
                concated_message.channels.append(channels)
                _, last_index = self.concated['index'].popleft()
            concated_message.length = last_index-begin_index
            return concated_message
        return None 