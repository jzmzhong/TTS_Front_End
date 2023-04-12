import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from dp.model.model import Model
from dp.model.utils import _trim_util_stop
from dp.preprocessing.text import Preprocessor
from dp.training.dataset import new_dataloader
from dp.training.dataset_pretrain import new_dataloader_pretrain
from dp.training.decorators import ignore_exception
from dp.training.losses import CrossEntropyLoss, CTCLoss
from dp.training.evaluation import evaluate_samples
from dp.utils.io import to_device, unpickle_binary


class Trainer:

    """ Performs model training. """

    def __init__(self, checkpoint_dir: Path, config: Dict[str, Any], loss_type='ctc') -> None:
        """
        Initializes a Trainer object.

        Args:
          checkpoint_dir (Path): Directory to store the model checkpoints.
          loss_type (str): Type of loss: 'ctc' for forward transformer models
                           and 'cross_entropy' for autoregressive models and GBERT.
        """

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))
        self.loss_type = loss_type
        if loss_type == 'ctc':
            if 'char_repeats' in config['model']:
                self.criterion = CTCLoss(config['model']['char_repeats'])
            else:
                self.criterion = CTCLoss()
        elif loss_type == 'cross_entropy':
            self.criterion = CrossEntropyLoss()
        else:
            raise ValueError(f'Loss not supported: {loss_type}')
        print(self.criterion)

    def train(self,
              model: Model,
              checkpoint: Dict[str, Any],
              store_phoneme_dict_in_model: bool = True) -> None:
        """
        Performs training of a transformer model.

        Args:
          model (Model): Model to be trained (can be a fresh model or restored from a checkpoint).
          checkpoint (Dict[str, Any]): Dictionary with entries 'optimizer': optimizer state dict,
                                       'preprocessor': Preprocessor and 'config': Dict.
          store_phoneme_dict_in_model (bool): Whether to store a dictionary of word-phoneme mappings
                                              in the model checkpoint so that it can be automatically
                                              loaded by a Phonemizer object.

        Returns:
          None: the checkpoints will be stored in a folder provided when instantiating a Trainer.
        """

        config = checkpoint['config']
        data_dir = Path(config['paths']['data_dir'])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Training on device:", device)
        model = model.to(device)
        model.train()

        criterion = self.criterion.to(device)

        if 'learning_rate_encoder_ratio' in config['training']:
            encoder_lr_ratio = float(config['training']['learning_rate_encoder_ratio'])
            base_lr = float(config['training']['learning_rate'])
            param_group_pretrained, param_group_new = [], []
            param_group_pretrained_names, param_group_new_names = [], []
            for param_name, param in model.named_parameters():
                if param_name.startswith("encoder_embedding.") or param_name.startswith("pos_encoder.") or param_name.startswith("transformer.encoder."):
                    param_group_pretrained.append(param)
                    param_group_pretrained_names.append(param_name)
                else:
                    param_group_new.append(param)
                    param_group_new_names.append(param_name)
            print("Pretrained Layers that have lower LR: " + str(param_group_pretrained_names))
            print("Newly Randomized Layers that have base LR: " + str(param_group_new_names))

            # Using two separate optimizers are preferred.

            # optimizer = Adam([{'params': param_group_pretrained, 'lr': encoder_lr_ratio*base_lr}, \
            #                  {'params': param_group_new}], \
            #                  lr=base_lr)

            optimizer_pretrained = Adam(param_group_pretrained)
            if 'optimizer_pretrained' in checkpoint:
                optimizer_pretrained.load_state_dict(checkpoint['optimizer_pretrained'])
            for g in optimizer_pretrained.param_groups:
                g['lr'] = encoder_lr_ratio*base_lr
            
            optimizer = Adam(param_group_new)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = base_lr
            
        else:
            optimizer = Adam(model.parameters())
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = config['training']['learning_rate']
            
            optimizer_pretrained = None

        if config['model']['type'] == "GBERT":
            train_loader = new_dataloader_pretrain(dataset_file=data_dir / 'train_dataset.pkl',
                                        drop_last=True, batch_size=config['training']['batch_size'],
                                        unk_idx=checkpoint['preprocessor'].text_tokenizer.token_to_idx["*"],
                                        non_special_indices=checkpoint['preprocessor'].text_tokenizer.non_special_indices,
                                        mask_ratio=config['model']['mask_ratio'],
                                        mask_UNK=config['model']['mask_UNK'],
                                        mask_random=config['model']['mask_random'],
                                        mask_original=config['model']['mask_original'])
            val_loader = new_dataloader_pretrain(dataset_file=data_dir / 'val_dataset.pkl',
                                        drop_last=False, batch_size=config['training']['batch_size_val'],
                                        unk_idx=checkpoint['preprocessor'].text_tokenizer.token_to_idx["*"],
                                        non_special_indices=checkpoint['preprocessor'].text_tokenizer.non_special_indices,
                                        mask_ratio="EVALUATE",
                                        mask_UNK=1.0,
                                        mask_random=0.0,
                                        mask_original=0.0)
        else:
            train_loader = new_dataloader(dataset_file=data_dir / 'train_dataset.pkl',
                                        drop_last=True, batch_size=config['training']['batch_size'])
            val_loader = new_dataloader(dataset_file=data_dir / 'val_dataset.pkl',
                                        drop_last=False, batch_size=config['training']['batch_size_val'])
            if store_phoneme_dict_in_model:
                phoneme_dict = unpickle_binary(data_dir / 'phoneme_dict.pkl')
                checkpoint['phoneme_dict'] = phoneme_dict

        val_batches = sorted([b for b in val_loader], key=lambda x: -x['text_len'][0])
        
        if optimizer_pretrained is not None:
            scheduler_pretrained = ReduceLROnPlateau(optimizer_pretrained,
                                        factor=config['training']['scheduler_plateau_factor'],
                                        patience=config['training']['scheduler_plateau_patience'],
                                        mode='min')
        scheduler = ReduceLROnPlateau(optimizer,
                                    factor=config['training']['scheduler_plateau_factor'],
                                    patience=config['training']['scheduler_plateau_patience'],
                                    mode='min')
        losses = []
        best_result = math.inf
        if 'step' not in checkpoint:
            checkpoint['step'] = 0
        start_epoch = checkpoint['step'] // len(train_loader)

        for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
            pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
            for i, batch in pbar:
                checkpoint['step'] += 1
                step = checkpoint['step']
                if optimizer_pretrained is not None:
                    self._set_warmup_diff_lr(model=model, optimizers=[optimizer, optimizer_pretrained], step=step,
                                        config=config)
                else:
                    self._set_warmup_lr(model=model, optimizer=optimizer, step=step,
                                    config=config)
                batch = to_device(batch, device)
                avg_loss = sum(losses) / len(losses) if len(losses) > 0 else math.inf
                pbar.set_description(desc=f'Epoch: {epoch} | Step {step} '
                                          f'| Loss: {avg_loss:#.4}', refresh=True)
                pred = model(batch)
                loss = criterion(pred, batch)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    optimizer.zero_grad()
                    if optimizer_pretrained is not None:
                        optimizer_pretrained.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if optimizer_pretrained is not None:
                        optimizer_pretrained.step()
                    losses.append(loss.item())

                self.writer.add_scalar('Loss/train', loss.item(), global_step=step)
                self.writer.add_scalar('Params/batch_size', config['training']['batch_size'],
                                       global_step=step)
                self.writer.add_scalar('Params/learning_rate', [g['lr'] for g in optimizer.param_groups][0],
                                       global_step=step)
                if optimizer_pretrained is not None:
                    self.writer.add_scalar('Params/learning_rate_pretrained', [g['lr'] for g in optimizer_pretrained.param_groups][0],
                                        global_step=step)

                if step % config['training']['validate_steps'] == 0:
                    val_loss = self._validate(model, val_batches)
                    self.writer.add_scalar('Loss/val', val_loss, global_step=step)
                    print(f'Epoch: {epoch} | Step {step} | Valid Loss: {avg_loss:#.4}')

                if step % config['training']['generate_steps'] == 0:
                    lang_samples = self._generate_samples(model=model,
                                                          preprocessor=checkpoint['preprocessor'],
                                                          val_batches=val_batches)
                    if config['model']['type'] == "GBERT":
                        eval_result = evaluate_samples(lang_samples=lang_samples, flag_pretrain=True)
                    else:
                        eval_result = evaluate_samples(lang_samples=lang_samples, flag_pretrain=False)
                    self._write_summaries(lang_samples=lang_samples,
                                          eval_result=eval_result,
                                          n_generate_samples=config['training']['n_generate_samples'],
                                          step=step)
                    if config['model']['type'] == "GBERT":
                        mean_mer = eval_result['mean_mer']
                        print(f'Epoch: {epoch} | Step {step} | Mean Mask Error Rate: {mean_mer}')
                        if eval_result['mean_mer'] is not None and eval_result['mean_mer'] < best_result:
                            self._save_model(model=model, optimizer=optimizer, optimizer_pretrained=None, checkpoint=checkpoint,
                                            path=self.checkpoint_dir / f'best_model.pt')
                            self._save_model(model=model, optimizer=None, optimizer_pretrained=None, checkpoint=checkpoint,
                                            path=self.checkpoint_dir / f'best_model_no_optim.pt')
                            best_result = eval_result['mean_mer']
                            print(f'Achieving better result (mean_mer): {best_result:#.4}')
                        scheduler.step(eval_result['mean_mer'])
                    else:
                        mean_per, mean_wer = eval_result['mean_per'], eval_result['mean_wer']
                        print(f'Epoch: {epoch} | Step {step} | Mean PER: {mean_per} | Mean WER: {mean_wer}')
                        if eval_result['mean_per'] is not None and eval_result['mean_per'] < best_result:
                            self._save_model(model=model, optimizer=optimizer, optimizer_pretrained=optimizer_pretrained, checkpoint=checkpoint,
                                            path=self.checkpoint_dir / f'best_model.pt')
                            self._save_model(model=model, optimizer=None, optimizer_pretrained=None, checkpoint=checkpoint,
                                            path=self.checkpoint_dir / f'best_model_no_optim.pt')
                            best_result = eval_result['mean_per']
                            print(f'Achieving better result (mean_per): {best_result:#.4}')
                        scheduler.step(eval_result['mean_per'])
                        if optimizer_pretrained is not None:
                            scheduler_pretrained.step(eval_result['mean_per'])

                if step % config['training']['checkpoint_steps'] == 0:
                    step = step // 1000
                    self._save_model(model=model, optimizer=optimizer, optimizer_pretrained=optimizer_pretrained, checkpoint=checkpoint,
                                     path=self.checkpoint_dir / f'model_step_{step}k.pt')

            losses = []
            print(f'Epoch: {epoch} | Step {step} | Train Loss: {avg_loss:#.4}')
            self._save_model(model=model, optimizer=optimizer, optimizer_pretrained=optimizer_pretrained, checkpoint=checkpoint,
                             path=self.checkpoint_dir / 'latest_model.pt')
            # randomly re-mask
            if config['model']['type'] == "GBERT" and epoch % config['training']['randomly_remask_epochs'] == 0:
                train_loader = new_dataloader_pretrain(dataset_file=data_dir / 'train_dataset.pkl',
                                    drop_last=True, batch_size=config['training']['batch_size'],
                                    unk_idx=checkpoint['preprocessor'].text_tokenizer.token_to_idx["*"],
                                    non_special_indices=checkpoint['preprocessor'].text_tokenizer.non_special_indices,
                                    mask_ratio=config['model']['mask_ratio'],
                                    mask_UNK=config['model']['mask_UNK'],
                                    mask_random=config['model']['mask_random'],
                                    mask_original=config['model']['mask_original'])


    def _validate(self, model: Model, val_batches: List[dict]) -> float:
        device = next(model.parameters()).device
        criterion = self.criterion.to(device)
        model.eval()
        val_losses = []
        for batch in val_batches:
            batch = to_device(batch, device)
            with torch.no_grad():
                pred = model(batch)
                loss = criterion(pred, batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())
        model.train()
        return sum(val_losses) / len(val_losses)

    @ignore_exception
    def _generate_samples(self,
                          model: Model,
                          preprocessor: Preprocessor,
                          val_batches: List[dict]) -> Dict[str, List[Tuple[List[str], List[str], List[str]]]]:

        """ Returns a dictionary with entries lang: Tuple of (word, generated, target) """

        device = next(model.parameters()).device
        model.eval()
        text_tokenizer = preprocessor.text_tokenizer
        phoneme_tokenizer = preprocessor.phoneme_tokenizer
        lang_tokenizer = preprocessor.lang_tokenizer
        lang_prediction_result = dict()

        for batch in val_batches:
            batch = to_device(batch, device)
            generated_batch, _ = model.generate(batch)
            for i in range(batch['text'].size(0)):
                text_len = batch['text_len'][i]
                text = batch['text'][i, :text_len]
                target = batch['phonemes'][i, :]
                lang = batch['language'][i]
                lang = lang_tokenizer.decode(lang.detach().cpu().item())
                generated = generated_batch[i, :].cpu()
                generated = _trim_util_stop(generated, phoneme_tokenizer.end_index)
                text, target = text.detach().cpu(), target.detach().cpu()
                text = text_tokenizer.decode(text, remove_special_tokens=True)
                generated = phoneme_tokenizer.decode(generated, remove_special_tokens=True)
                target = phoneme_tokenizer.decode(target, remove_special_tokens=True)
                lang_prediction_result[lang] = lang_prediction_result.get(lang, []) + [(text, generated, target)]

        model.train()

        return lang_prediction_result

    @ignore_exception
    def _write_summaries(self,
                         lang_samples: Dict[str, List[Tuple[List[str], List[str], List[str]]]],
                         eval_result: Dict[str, Any],
                         n_generate_samples: int,
                         step: int) -> None:
        if "mean_per" not in eval_result:
            self.writer.add_scalar(f'Mask_Prediction_Error_Rate/mean',
                               eval_result['mean_mer'], global_step=step)
        else:
            self.writer.add_scalar(f'Phoneme_Error_Rate/mean',
                               eval_result['mean_per'], global_step=step)
            self.writer.add_scalar(f'Word_Error_Rate/mean',
                               eval_result['mean_wer'], global_step=step)

        for lang in lang_samples.keys():
            result = eval_result[lang]
            if "per" not in result:
                self.writer.add_scalar(f'Mask_Prediction_Error_Rate/{lang}',
                               result['mer'], global_step=step)
            else:
                self.writer.add_scalar(f'Phoneme_Error_Rate/{lang}',
                                    result['per'], global_step=step)
                self.writer.add_scalar(f'Word_Error_Rate/{lang}',
                                    result['wer'], global_step=step)

        for lang, samples in lang_samples.items():
            samples = [(''.join(w), ''.join(p), ''.join(t)) for w, p, t in samples]
            word_counts = Counter([word for word, _, _ in samples])
            samples_dedup = [(w, p, t) for w, p, t in samples if word_counts[w] == 1]
            log_texts = dict()
            for word, pred, target in samples_dedup:
                log_texts[word] = f'     {word:<30} {pred:<30} {target:<30}'
            log_text_items = sorted(log_texts.items(), key=lambda x: -len(x[0]))
            log_text_list = [v for k, v in log_text_items]
            log_text = '\n'.join(log_text_list[:n_generate_samples])
            self.writer.add_text(f'{lang}/text_prediction_target', log_text, global_step=step)

    def _save_model(self,
                    model: torch.nn.Module,
                    optimizer: torch.optim,
                    optimizer_pretrained: torch.optim,
                    checkpoint: Dict[str, Any],
                    path: Path) -> None:
        checkpoint['model'] = model.state_dict()
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        else:
            checkpoint['optimizer'] = None
        if optimizer_pretrained is not None:
            checkpoint['optimizer_pretrained'] = optimizer_pretrained.state_dict()
        else:
            checkpoint['optimizer'] = None
        torch.save(checkpoint, str(path))

    def _set_warmup_lr(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim,
                       step: int,
                       config: Dict[str, Any]) -> None:

        warmup_steps = config['training']['warmup_steps']
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = 1.0 - max(warmup_steps - step, 0) / warmup_steps
            if 'learning_rate_encoder_ratio' in config['training']:
                encoder_lr_ratio = float(config['training']['learning_rate_encoder_ratio'])
                base_lr = float(config['training']['learning_rate'])
                optimizer.param_groups[0]['lr'] = base_lr * warmup_factor * encoder_lr_ratio # pretrained layers
                optimizer.param_groups[1]['lr'] = base_lr * warmup_factor # newly randomized layers
            else:
                for g in optimizer.param_groups:
                    g['lr'] = config['training']['learning_rate'] * warmup_factor
    
    def _set_warmup_diff_lr(self,
                       model: torch.nn.Module,
                       optimizers: List,
                       step: int,
                       config: Dict[str, Any]) -> None:

        warmup_steps = config['training']['warmup_steps']
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = 1.0 - max(warmup_steps - step, 0) / warmup_steps
            
            assert 'learning_rate_encoder_ratio' in config['training']
            assert len(optimizers) == 2

            encoder_lr_ratio = float(config['training']['learning_rate_encoder_ratio'])
            base_lr = float(config['training']['learning_rate'])
            
            # optimizer.param_groups[0]['lr'] = base_lr * warmup_factor * encoder_lr_ratio # pretrained layers
            # optimizer.param_groups[1]['lr'] = base_lr * warmup_factor # newly randomized layers
            
            for g in optimizers[0].param_groups:
                g['lr'] = base_lr * warmup_factor
            for g in optimizers[1].param_groups:
                g['lr'] = base_lr * warmup_factor * encoder_lr_ratio
