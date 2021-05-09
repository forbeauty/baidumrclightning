import pandas as pd
import pytorch_lightning as pl
import torch
import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):

    def __init__(self, args, split, tokenizer):

        self.args = args
        examples = []
        if split == 'train':
            with open(self.args.config['dataset']['train_path'], "r", encoding="utf8") as f:
                input_data = json.load(f)["data"]
        elif split == 'val':
            with open(self.args.config['dataset']['dev_path'], "r", encoding="utf8") as f:
                input_data = json.load(f)["data"]
        elif split == 'test':
            with open(self.args.config['dataset']['test_path'], "r", encoding="utf8") as f:
                input_data = json.load(f)["data"]
        else:
            raise ValueError

        for entry in input_data:
            #     title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                title = paragraph["title"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = []
                    answers = []
                    is_impossible = False

                    if "is_impossible" in qa.keys():
                        is_impossible = qa["is_impossible"]

                    answer_starts = [answer["answer_start"] for answer in qa.get("answers", [])]
                    answers = [answer["text"].strip() for answer in qa.get("answers", [])]

                    examples.append({
                        "id": qas_id,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "answer_starts": answer_starts,
                        "is_impossible": is_impossible
                    })
        if args.fold > -1 and split == 'train':
            single_len = len(examples) // args.config['solver']['kfold']
            single_fold_len = single_len * (args.config['solver']['kfold'] - 1)
            examples.extend(examples)
            examples = examples[args.fold * single_len: args.fold * single_len + single_fold_len]

        if self.args.overfit:
            examples = examples[:4]

        # questions = [examples[i]['question'] for i in range(len(examples))]
        questions_title = [examples[i]['question'] + examples[i]['title'] for i in range(len(examples))]
        # title_contexts = [examples[i]['title'] + examples[i]['context'] for i in range(len(examples))]
        contexts = [examples[i]['context'] for i in range(len(examples))]

        tokenized_examples = tokenizer(text=questions_title,
                                       text_pair=contexts,
                                       padding="max_length",
                                       max_length=args.config['solver']['max_length'],
                                       truncation="only_second",
                                       stride=args.config['solver']['stride'],
                                       return_offsets_mapping=True,
                                       return_overflowing_tokens=True
                                       )
        df_tmp = pd.DataFrame.from_dict(tokenized_examples, orient="index").T
        tokenized_examples = df_tmp.to_dict(orient="records")

        for i, tokenized_example in enumerate(tokenized_examples):
            input_ids = tokenized_example["input_ids"]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            offsets = tokenized_example['offset_mapping']
            sequence_ids = tokenized_example['token_type_ids']

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_example['overflow_to_sample_mapping']
            answers = examples[sample_index]['answers']
            answer_starts = examples[sample_index]['answer_starts']

            # If no answers are given, set the cls_index as answer.
            if len(answer_starts) == 0 or (answer_starts[0] == -1):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_starts[0]
                end_char = start_char + len(answers[0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 2
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and
                        offsets[token_end_index][1] >= end_char):
                    tokenized_examples[i]["start_positions"] = cls_index
                    tokenized_examples[i]["end_positions"] = cls_index
                    tokenized_examples[i]['answerable_label'] = 0
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples[i]["start_positions"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples[i]["end_positions"] = token_end_index + 1
                    tokenized_examples[i]['answerable_label'] = 1

            # evaluate的时候有用
            tokenized_examples[i]["example_id"] = examples[sample_index]['id']
            tokenized_examples[i]["offset_mapping"] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_example["offset_mapping"])
            ]

        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


class MRCDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.config['model']['name'])
        # if not os.path.exists(self.args.config['dataset']['total_train_path']):
        #     with open(os.path.join(os.getcwd(), self.args.config['dataset']['train_path']), mode='r', encoding='utf-8') as f1, \
        #             open(os.path.join(os.getcwd(), self.args.config['dataset']['dev_path']), mode='r', encoding='utf-8') as f2:
        #         json.load(f1)['data'][0]['paragraphs'].update(json.load(f2)['data'][0]['paragraphs'])
        #         json.dump(f1, open(os.path.join(os.getcwd(), self.args.config['dataset']['total_train_path'])))

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:

            self.train_dataset = BaseDataset(
                args=self.args,
                split='train',
                tokenizer=self.tokenizer
            )

            if self.args.config['solver']['val_check_interval'] > 0:
                self.val_dataset = BaseDataset(
                    args=self.args,
                    split='val',
                    tokenizer=self.tokenizer
                )

        if stage == 'test' or stage is None:

            self.test_dataset = BaseDataset(
                args=self.args,
                split='test',
                tokenizer=self.tokenizer
            )


    def prepare_data(self):
        pass

    def train_dataloader(self):

        return DataLoader(self.train_dataset,
                          batch_size=self.args.config['solver']['batch_size'],
                          # shuffle=True,
                          collate_fn=self.collate_fn,
                          # num_workers=0,
                          # prefetch_factor=4,
                          # persistent_workers=True
                          )

    def val_dataloader(self):

        return DataLoader(self.val_dataset,
                          batch_size=self.args.config['solver']['batch_size'],
                          # shuffle=False,
                          collate_fn=self.collate_fn,
                          # num_workers=0,
                          # prefetch_factor=4,
                          # persistent_workers=True
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.args.config['solver']['batch_size'],
                          # shuffle=False,
                          collate_fn=self.collate_fn,
                          # num_workers=0,
                          # prefetch_factor=4,
                          # persistent_workers=True
                          )

    def collate_fn(self, batch):
        # max_len = max([sum(x['attention_mask']) for x in batch])
        max_len = self.args.config['solver']['max_length']
        all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
        all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
        all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
        all_answerable_label = torch.tensor([x["answerable_label"] for x in batch])
        all_start_positions = torch.tensor([x["start_positions"] for x in batch])
        all_end_positions = torch.tensor([x["end_positions"] for x in batch])

        return {
            "all_input_ids": all_input_ids,
            "all_token_type_ids": all_token_type_ids,
            "all_attention_mask": all_attention_mask,
            "all_start_positions": all_start_positions,
            "all_end_positions": all_end_positions,
            "all_answerable_label": all_answerable_label,
        }

