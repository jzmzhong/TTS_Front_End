from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence

from dp.model.utils import get_dedup_tokens, _make_len_mask, _generate_square_subsequent_mask, PositionalEncoding
from dp.preprocessing.text import Preprocessor


class ModelType(Enum):
    GBERT = 'GBERT'
    TRANSFORMER = 'transformer'
    TRANSFORMER_TRIMMED = 'transformer_trimmed'
    TRANSFORMER_TRIMMED_GBERT_FINETUNE = 'transformer_trimmed_GBERT_finetune'
    TRANSFORMER_ALIGNED = 'transformer_aligned'
    AUTOREG_TRANSFORMER = 'autoreg_transformer'
    AUTOREG_TRANSFORMER_GBERT_FINETUNE = 'autoreg_transformer_GBERT_finetune'

    def require_cross_entropy_loss(self) -> bool:
        """
        Returns: bool: Whether the model requires cross entropy loss.
        """
        return self in {ModelType.GBERT, ModelType.AUTOREG_TRANSFORMER, ModelType.AUTOREG_TRANSFORMER_GBERT_FINETUNE, ModelType.TRANSFORMER_ALIGNED}


class Model(torch.nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates phonemes for a text batch

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing 'text' (tokenized text tensor),
                       'text_len' (text length tensor),
                       'start_index' (phoneme start indices for AutoregressiveTransformer)

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: The predictions. The first element is a tensor (phoneme tokens)
          and the second element  is a tensor (phoneme token probabilities)
        """
        pass


class GBERT(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=2048,
                 layers=6,
                 dropout=0.1,
                 heads=8) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self,
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:         # shape: [N, T]
        """
        Forward pass of the model on a data batch.

        Args:
         batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

        Returns:
          Tensor: Predictions.
        """

        x = batch['text']
        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x[:-1,:,:])
        x = x.transpose(0, 1)
        return x

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = self.forward(batch)
        
        logits_batch = x.softmax(-1)
        out_tokens, out_probs = [], []
        for i in range(logits_batch.size(0)):
            logits = logits_batch[i]
            max_logits, max_indices = torch.max(logits, dim=-1)
            max_logits = max_logits[max_indices!=0]
            max_indices = max_indices[max_indices!=0]
            out_tokens.append(max_indices)
            out_probs.append(max_logits)
        out_tokens = pad_sequence(out_tokens, batch_first=True, padding_value=0.).long()
        out_probs = pad_sequence(out_probs, batch_first=True, padding_value=0.)
        
        return out_tokens, out_probs

    @classmethod
    def from_config(cls, config: dict) -> 'GBERT':
        preprocessor = Preprocessor.from_config(config)
        return GBERT(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


class ForwardTransformer(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self,
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:         # shape: [N, T]
        """
        Forward pass of the model on a data batch.

        Args:
         batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

        Returns:
          Tensor: Predictions.
        """

        x = batch['text']
        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

        Returns:
          Tuple: The first element is a Tensor (phoneme tokens) and the second element
                 is a tensor (phoneme token probabilities).
        """

        with torch.no_grad():
            x = self.forward(batch)
        tokens, logits = get_dedup_tokens(x)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> 'ForwardTransformer':
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


class ForwardTransformerTrimmed(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1,
                 char_repeats=3) -> None:
        super().__init__()

        self.char_repeats = char_repeats
        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder_1 = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder_1 = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers-1,
                                          norm=encoder_norm)
        self.pos_encoder_2 = PositionalEncoding(d_model, dropout)
        self.encoder_2 = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=1,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self,
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:         # shape: [N, T]
        """
        Forward pass of the model on a data batch.

        Args:
         batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

        Returns:
          Tensor: Predictions.
        """

        x = batch['text']
        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder_1(x)
        x = self.encoder_1(x, src_key_padding_mask=src_pad_mask)
        x = torch.cat((x[:1,:,:], x[1:-1,:,:].repeat_interleave(self.char_repeats, 0), x[-1:,:,:]), dim=0)
        src_pad_mask = torch.cat((src_pad_mask[:,:1], src_pad_mask[:,1:-1].repeat_interleave(self.char_repeats, 1), src_pad_mask[:,-1:]), dim=1)
        x = self.pos_encoder_2(x)
        x = self.encoder_2(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

        Returns:
          Tuple: The first element is a Tensor (phoneme tokens) and the second element
                 is a tensor (phoneme token probabilities).
        """

        with torch.no_grad():
            x = self.forward(batch)
        tokens, logits = get_dedup_tokens(x)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> 'ForwardTransformerTrimmed':
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformerTrimmed(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads'],
            char_repeats=config['model']['char_repeats']
        )


class ForwardTransformerAligned(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self,
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:         # shape: [N, T]
        """
        Forward pass of the model on a data batch.

        Args:
         batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

        Returns:
          Tensor: Predictions.
        """

        x = batch['text']
        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x[:-1,:,:])
        x = x.transpose(0, 1)
        return x

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

        Returns:
          Tuple: The first element is a Tensor (phoneme tokens) and the second element
                 is a tensor (phoneme token probabilities).
        """

        with torch.no_grad():
            x = self.forward(batch)
            # x = x[-1:, :, :]
        logits_batch = x.softmax(-1)
        out_tokens, out_probs = [], []
        for i in range(logits_batch.size(0)):
            logits = logits_batch[i]
            max_logits, max_indices = torch.max(logits, dim=-1)
            max_logits = max_logits[max_indices!=0]
            max_indices = max_indices[max_indices!=0]
            out_tokens.append(max_indices)
            out_probs.append(max_logits)
        tokens = pad_sequence(out_tokens, batch_first=True, padding_value=0.).long()
        logits = pad_sequence(out_probs, batch_first=True, padding_value=0.)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> 'ForwardTransformer':
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformerAligned(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


class AutoregressiveTransformer(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 end_index: int,
                 d_model=512,
                 d_fft=1024,
                 encoder_layers=4,
                 decoder_layers=4,
                 dropout=0.1,
                 heads=1):
        super().__init__()

        self.end_index = end_index
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers, dim_feedforward=d_fft,
                                          dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, batch: Dict[str, torch.Tensor]):         # shape: [N, T]
        """
        Foward pass of the model on a data batch.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entries 'text' (text tensor) and 'phonemes'
                                           (phoneme tensor for teacher forcing).

        Returns:
          Tensor: Predictions.
        """

        src = batch['text']
        trg = batch['phonemes'][:, :-1]

        src = src.transpose(0, 1)        # shape: [T, N]
        trg = trg.transpose(0, 1)

        trg_mask = _generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = _make_len_mask(src).to(trg.device)
        trg_pad_mask = _make_len_mask(trg).to(trg.device)

        src = self.encoder_embedding(src)
        src = self.pos_encoder(src)

        trg = self.decoder_embedding(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=None, tgt_mask=trg_mask,
                                  memory_mask=None, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)
        output = output.transpose(0, 1)
        return output

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor],
                 max_len: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing the input to the model with entries 'text'
                                           and 'start_index'
          max_len (int): Max steps of the autoregressive inference loop.

        Returns:
          Tuple: Predictions. The first element is a Tensor of phoneme tokens and the second element
                 is a Tensor of phoneme token probabilities.
        """

        input = batch['text']
        start_index = batch['start_index']

        batch_size = input.size(0)
        input = input.transpose(0, 1)          # shape: [T, N]
        src_pad_mask = _make_len_mask(input).to(input.device)
        with torch.no_grad():
            input = self.encoder_embedding(input)
            input = self.pos_encoder(input)
            input = self.transformer.encoder(input,
                                             src_key_padding_mask=src_pad_mask)
            out_indices = start_index.unsqueeze(0)
            out_logits = []
            for i in range(max_len):
                tgt_mask = _generate_square_subsequent_mask(i + 1).to(input.device)
                output = self.decoder_embedding(out_indices)
                output = self.pos_decoder(output)
                output = self.transformer.decoder(output,
                                                  input,
                                                  memory_key_padding_mask=src_pad_mask,
                                                  tgt_mask=tgt_mask)
                output = self.fc_out(output)  # shape: [T, N, V]
                out_tokens = output.argmax(2)[-1:, :]
                out_logits.append(output[-1:, :, :])

                out_indices = torch.cat([out_indices, out_tokens], dim=0)
                stop_rows, _ = torch.max(out_indices == self.end_index, dim=0)
                if torch.sum(stop_rows) == batch_size:
                    break

        out_indices = out_indices.transpose(0, 1)  # out shape [N, T]
        out_logits = torch.cat(out_logits, dim=0).transpose(0, 1) # out shape [N, T, V]
        out_logits = out_logits.softmax(-1)
        out_probs = torch.ones((out_indices.size(0), out_indices.size(1)))
        for i in range(out_indices.size(0)):
            for j in range(0, out_indices.size(1)-1):
                out_probs[i, j+1] = out_logits[i, j].max()
        return out_indices, out_probs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoregressiveTransformer':
        """
        Initializes an autoregressive Transformer model from a config.
        Args:
          config (dict): Configuration containing the hyperparams.

        Returns:
          AutoregressiveTransformer: Model object.
        """

        preprocessor = Preprocessor.from_config(config)
        if 'encoder_layers' not in config['model']:
            config['model']['encoder_layers'] = config['model']['layers']
        if 'decoder_layers' not in config['model']:
            config['model']['decoder_layers'] = config['model']['layers']
        return AutoregressiveTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            end_index=preprocessor.phoneme_tokenizer.end_index,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            encoder_layers=config['model']['encoder_layers'],
            decoder_layers=config['model']['decoder_layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


def create_model(model_type: ModelType, config: Dict[str, Any]) -> Model:
    """
    Initializes a model from a config for a given model type.

    Args:
        model_type (ModelType): Type of model to be initialized.
        config (dict): Configuration containing hyperparams.

    Returns: Model: Model object.
    """

    if model_type is ModelType.GBERT:
        model = GBERT.from_config(config)
    elif model_type is ModelType.TRANSFORMER:
        model = ForwardTransformer.from_config(config)
    elif model_type in [ModelType.TRANSFORMER_TRIMMED, ModelType.TRANSFORMER_TRIMMED_GBERT_FINETUNE]:
        model = ForwardTransformerTrimmed.from_config(config)
    elif model_type is ModelType.TRANSFORMER_ALIGNED:
        model = ForwardTransformerAligned.from_config(config)
    elif model_type in [ModelType.AUTOREG_TRANSFORMER, ModelType.AUTOREG_TRANSFORMER_GBERT_FINETUNE]:
        model = AutoregressiveTransformer.from_config(config)
    else:
        raise ValueError(f'Unsupported model type: {model_type}. '
                         f'Supported types: {[t.value for t in ModelType]}')
    
    # Load GBERT to Encoder of Autoregressive G2P Model
    if model_type in [ModelType.AUTOREG_TRANSFORMER_GBERT_FINETUNE]:
        GBERT_checkpoint = torch.load(config['paths']['GBERT_path'])
        print(f"Loading GBERT from: {config['paths']['GBERT_path']}")
        # import pdb; pdb.set_trace()
        GBERT_checkpoint_processed = {}
        for k, v in GBERT_checkpoint['model'].items():
            if k.startswith("embedding."): # load encoder embedding
                GBERT_checkpoint_processed["encoder_" + k] = v
            elif k.startswith("encoder."): # load encoder
                GBERT_checkpoint_processed["transformer." + k] = v
            elif k.startswith("fc_out."): # ignore FC in GBERT
                continue
            else:
                GBERT_checkpoint_processed[k] = v
        model.load_state_dict(GBERT_checkpoint_processed, strict=False)
        model.eval()
    # Load GBERT to Encoder of Forward Trimmed G2P Model
    elif model_type in [ModelType.TRANSFORMER_TRIMMED_GBERT_FINETUNE]:
        GBERT_checkpoint = torch.load(config['paths']['GBERT_path'])
        print(f"Loading GBERT from: {config['paths']['GBERT_path']}")
        # import pdb; pdb.set_trace()
        GBERT_checkpoint_processed = {}
        for k, v in GBERT_checkpoint['model'].items():
            if k.startswith("encoder.layers."): # load encoder model.encoder_1.layers[0] encoder.layers.0
                src_layer = k.split(".")[2]
                if src_layer in config['model']['initialization_mapping']:
                    trg_layer = config['model']['initialization_mapping'][src_layer]
                else:
                    continue
                GBERT_checkpoint_processed[k.replace("encoder.layers."+src_layer, "encoder_1.layers."+trg_layer)] = v
            elif k.startswith("encoder.norm."):
                GBERT_checkpoint_processed[k.replace("encoder.norm.", "encoder_1.norm.")] = v
            elif k.startswith("pos_encoder."): # load positional encoding
                GBERT_checkpoint_processed[k.replace("pos_encoder.", "pos_encoder_1.")] = v
            elif k.startswith("fc_out."): # ignore FC in GBERT
                continue
            else:
                GBERT_checkpoint_processed[k] = v
        # switch to strict=True in debugging to see which layers are not initializedß
        model.load_state_dict(GBERT_checkpoint_processed, strict=False)
        model.eval()

    print(model)
    return model


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[Model, Dict[str, Any]]:
    """
    Initializes a model from a checkpoint (.pt file).

    Args:
        checkpoint_path (str): Path to checkpoint file (.pt).
        device (str): Device to put the model to ('cpu' or 'cuda').

    Returns: Tuple: The first element is a Model (the loaded model)
             and the second element is a dictionary (config).
    """

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['config']['model']['type']
    model_type = ModelType(model_type)
    model = create_model(model_type, config=checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint