import os
import torch
import copy
import pytorch_lightning as pl
import transformers
import torch.nn.functional as F
import json
from torch import nn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModel
from squad_metric import f1_em_metric
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from post_process_for_qa import postprocess_qa_predictions


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertlikeModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(self.args.config['model']['name'])
        self.pooler = Pooler(self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.classifier_cls = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = output['last_hidden_state']
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits


class CrossEntropyLossForQuestionAnswer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, label):

        start_logits, end_logits, cls_logits = pred
        start_position, end_position, answerable_label = label

        start_loss = F.cross_entropy(input=start_logits, target=start_position)
        end_loss = F.cross_entropy(input=end_logits, target=end_position)
        cls_loss = F.cross_entropy(input=cls_logits, target=answerable_label)

        mrc_loss = (start_loss + end_loss) / 2
        loss = (mrc_loss + cls_loss) / 2

        return loss, mrc_loss, cls_loss


class MRCModel(pl.LightningModule):

    def __init__(self, args, datamodule):

        super().__init__()
        self.args = args
        self.datamodule = datamodule
        self.emb_name = 'word_embeddings.weight'
        self.fgm_backup = {}
        self.pgd_emb_backup = {}
        self.pgd_grad_backup = {}
        self.pgd_alpha = 0.3
        self.pgd_epsilon = 1
        self.best_f1_thresh = {'f1': 0, 'best_f1_thresh': 0.5}

        self.model = BertlikeModel(self.args)

        self.optimizer_class = getattr(torch.optim, self.args.config['model']['optimizer'])
        self.scheduler_fn = getattr(transformers, self.args.config['model']['lr_schedule'])
        self.criterion_fn = CrossEntropyLossForQuestionAnswer()
        self.metric_fn = f1_em_metric
        if self.args.config['solver']['adversarial_training'] == 'fgm' or \
                self.args.config['solver']['adversarial_training'] == 'pgd':
            self.automatic_optimization = False

    def forward(self, input_ids, token_type_ids, attention_mask):

        return self.model(input_ids, token_type_ids, attention_mask)

    def training_step(self, batch, batch_idx):

        if self.args.config['solver']['adversarial_training'] == 'fgm':

            opt = self.optimizers()

            logits = self(
                input_ids=batch['all_input_ids'],
                token_type_ids=batch['all_token_type_ids'],
                attention_mask=batch['all_attention_mask']
            )

            loss, mrc_loss, cls_loss = self.criterion_fn(logits, (
                batch["all_start_positions"],
                batch["all_end_positions"],
                batch["all_answerable_label"]
            ))
            self.log('train_loss', loss, logger=True)
            opt.zero_grad()
            self.manual_backward(loss=loss)
            self.fgm_attack()
            logits_adv, mrc_loss, cls_loss = self(
                input_ids=batch['all_input_ids'],
                token_type_ids=batch['all_token_type_ids'],
                attention_mask=batch['all_attention_mask']
            )
            loss_adv = self.criterion_fn(logits_adv, (
                batch["all_start_positions"],
                batch["all_end_positions"],
                batch["all_answerable_label"]
            ))
            opt.zero_grad()  # self.zero_grad() will set all parameters to zero.
            self.manual_backward(loss=loss_adv)
            self.fgm_restore()
            clip_grad_norm_(self.model.parameters(), self.args.config['gradient_clip_val'], norm_type=2)
            opt.step()

        elif self.args.config['solver']['adversarial_training'] == 'pgd':

            opt = self.optimizers()

            K = 3
            loss_adv = 0
            logits = self(
                input_ids=batch['all_input_ids'],
                token_type_ids=batch['all_token_type_ids'],
                attention_mask=batch['all_attention_mask']
            )
            loss, mrc_loss, cls_loss = self.criterion_fn(logits, (
                batch["all_start_positions"],
                batch["all_end_positions"],
                batch["all_answerable_label"]
            ))
            self.log('train_loss', loss, logger=True)
            opt.zero_grad()
            self.manual_backward(loss)
            self.pgd_backup_grad()
            # 对抗训练
            for t in range(K):
                self.pgd_attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    opt.zero_grad()
                else:
                    self.pgd_restore_grad()
                logits_adv = self(
                    input_ids=batch['all_input_ids'],
                    token_type_ids=batch['all_token_type_ids'],
                    attention_mask=batch['all_attention_mask']
                )
                loss_adv, mrc_loss, cls_loss = self.criterion_fn(logits_adv, (
                    batch["all_start_positions"],
                    batch["all_end_positions"],
                    batch["all_answerable_label"]
                ))

                self.manual_backward(loss_adv)  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.pgd_restore()  # 恢复embedding参数
            # 梯度下降，更新参数
            clip_grad_norm_(self.model.parameters(), self.args.config['gradient_clip_val'], norm_type=2)
            opt.step()
            opt.zero_grad()

        else:

            logits = self(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            loss, mrc_loss, cls_loss = self.criterion_fn(logits, (
                batch["all_start_positions"],
                batch["all_end_positions"],
                batch["all_answerable_label"]
            ))
            self.log('train_loss', loss, logger=True)
            self.log('mrc_loss', mrc_loss, prog_bar=True, logger=True)
            self.log('cls_loss', cls_loss, prog_bar=True, logger=True)
            return loss

    # def training_step_end(self, outputs):
    #     pass

    def fgm_attack(self, epsilon=1):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.fgm_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def fgm_restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.fgm_backup
                param.data = self.fgm_backup[name]
        self.fgm_backup = {}

    def pgd_attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.pgd_emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.pgd_alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.pgd_project(name, param.data, self.pgd_epsilon)

    def pgd_restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.pgd_emb_backup
                param.data = self.pgd_emb_backup[name]
        self.pgd_emb_backup = {}

    def pgd_project(self, param_name, param_data, epsilon):
        r = param_data - self.pgd_emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.pgd_emb_backup[param_name] + r

    def pgd_backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.pgd_grad_backup[name] = param.grad.clone()

    def pgd_restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.pgd_grad_backup[name]

    def validation_step(self, batch, batch_idx):

        start_logits, end_logits, cls_logits = self(
            input_ids=batch['all_input_ids'],
            token_type_ids=batch['all_token_type_ids'],
            attention_mask=batch['all_attention_mask']
        )

        return {'start_logits': start_logits, 'end_logits': end_logits, 'cls_logits': cls_logits, 'batch': batch}

    def validation_epoch_end(self, outputs):

        all_start_logits = torch.cat([x['start_logits'] for x in outputs], dim=0)
        all_end_logits = torch.cat([x['end_logits'] for x in outputs], dim=0)
        all_cls_logits = torch.cat([x['cls_logits'] for x in outputs], dim=0)
        all_start_positions = torch.cat([x['batch']['all_start_positions'] for x in outputs], dim=0)
        all_end_positions = torch.cat([x['batch']['all_end_positions'] for x in outputs], dim=0)
        all_answerable_label = torch.cat([x['batch']['all_answerable_label'] for x in outputs], dim=0)

        loss, mrc_loss, cls_loss = self.criterion_fn(
            (
                all_start_logits,
                all_end_logits,
                all_cls_logits
            ),
            (
                all_start_positions,
                all_end_positions,
                all_answerable_label
            ))
        self.log('val_loss', loss, prog_bar=True, logger=True)

        all_start_logits = all_start_logits.detach().cpu().numpy()
        all_end_logits = all_end_logits.detach().cpu().numpy()
        all_cls_logits = all_cls_logits.detach().cpu().numpy()

        all_predictions = postprocess_qa_predictions(
            self.datamodule.val_dataset.examples,
            self.datamodule.val_dataset.tokenized_examples,
            (all_start_logits, all_end_logits, all_cls_logits),
            True,
            self.args.config['solver']['n_best_size'],
            self.args.config['solver']['max_answer_length'],
            self.args.config['solver']['cls_threshold']
        )

        f1_score, em_score, total_count, skip_count = self.metric_fn(self.datamodule.val_dataset.examples, all_predictions)
        if f1_score > self.best_f1_thresh['f1']:
            self.best_f1_thresh['f1'] = f1_score
            # best_model = copy.deepcopy(self.model.module if hasattr(self.model, "module") else self.model)
            # torch.save(best_model.state_dict(), os.path.join(self.model.args.save_path, "best_model.ckpt"))
            # self.best_f1_thresh['best_f1_thresh'] = evaluation['best_f1_thresh']
        evaluation = {'f1_score': f1_score,
            'em_score': em_score,
            'total_count': total_count,
            'skip_count': skip_count,
            'Val_mrc_loss': mrc_loss,
            'val_cls_loss': cls_loss,
            'val_loss': loss
            }

        self.args.logger.info(f"raw-f1={evaluation['f1_score']}  raw-em={evaluation['em_score']}  best_f1_thresh={self.best_f1_thresh['best_f1_thresh']}  "
                              f""f"best-f1={self.best_f1_thresh['f1']}  best-em={evaluation['em_score']}  val_loss={loss}")
        self.log_dict(evaluation, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):

        start_logits, end_logits, cls_logits = self(
            input_ids=batch['all_input_ids'],
            token_type_ids=batch['all_token_type_ids'],
            attention_mask=batch['all_attention_mask']
        )

        return {'start_logits': start_logits, 'end_logits': end_logits, 'cls_logits': cls_logits}

    def test_epoch_end(self, outputs):

        os.makedirs(self.args.save_path, exist_ok=True)

        all_start_logits = torch.cat([x['start_logits'] for x in outputs], dim=0)
        all_end_logits = torch.cat([x['end_logits'] for x in outputs], dim=0)
        all_cls_logits = torch.cat([x['cls_logits'] for x in outputs], dim=0)

        all_start_logits = all_start_logits.detach().cpu().numpy()
        all_end_logits = all_end_logits.detach().cpu().numpy()
        all_cls_logits = all_cls_logits.detach().cpu().numpy()
        all_predictions = postprocess_qa_predictions(
            self.datamodule.test_dataset.examples,
            self.datamodule.test_dataset.tokenized_examples,
            (all_start_logits, all_end_logits, all_cls_logits),
            True,
            self.args.config['solver']['n_best_size'],
            self.args.config['solver']['max_answer_length'],
            self.args.config['solver']['cls_threshold']
        )

        with open(os.path.join(self.args.save_path, 'result.json'), "w", encoding='utf-8') as writer:
            writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    def setup(self, stage=None):

        if stage == 'fit' or stage == None:
            # Calculate total steps
            self.total_steps = (
                    (len(self.train_dataloader()) // max(1, len(self.args.gpu) if self.args.gpu else 0,
                                                         self.args.tpu if self.args.tpu else 0))
                    // self.args.config['solver']['accumulate_grad_batches']
                    * float(self.args.config['solver']['num_epochs'])
            )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = list(self.model.bert.named_parameters())
        linear_params = list(self.model.classifier.named_parameters())
        linear_params.extend(list(self.model.classifier_cls.named_parameters()))
        grouped_parameters = [
            {
                'params': [
                    p for n, p in params
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.args.config['solver']['weight_decay'],
                'lr': self.args.config['solver']['initial_lr']
            },
            {
                'params': [
                    p for n, p in params
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
                'lr': self.args.config['solver']['initial_lr']
            },
            {
                'params': [
                    p for n, p in linear_params
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.args.config['solver']['weight_decay'],
                'lr': self.args.config['solver']['linear_initial_lr']
            },
            {
                'params': [
                    p for n, p in linear_params
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
                'lr': self.args.config['solver']['linear_initial_lr']
            }

        ]

        warmup_steps = self.args.config['solver']['warmup_fraction'] * self.total_steps
        optimizer = self.optimizer_class(grouped_parameters,
                                         lr=self.args.config['solver']['initial_lr'],
                                         weight_decay=self.args.config['solver']['weight_decay'])
        if self.model.args.config['model']['lr_schedule'] == 'get_linear_schedule_with_warmup':
            scheduler = self.scheduler_fn(optimizer=optimizer,
                                        num_warmup_steps=warmup_steps,
                                        num_training_steps=self.total_steps)
        elif self.model.args.config['model']['lr_schedule'] == 'get_constant_schedule':
            scheduler = self.scheduler_fn(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):

        # early_stop_callback = EarlyStopping(
        #     monitor='f1' if self.args.config['solver']['val_check_interval'] > 0 else 'train_loss',
        #     patience=3,
        #     verbose=False,
        #     mode='max' if self.args.config['solver']['val_check_interval'] > 0 else 'min',
        # )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args.save_path,
            filename=f"{{epoch}}___{{{'f1_score' if self.args.config['solver']['val_check_interval'] > 0 else 'train_loss'}:.4f}}",
            save_top_k=3,
            verbose=False,
            monitor='f1_score' if self.args.config['solver']['val_check_interval'] > 0 else 'train_loss',
            mode='max' if self.args.config['solver']['val_check_interval'] > 0 else 'min',
            prefix=''
        )


        return [checkpoint_callback]