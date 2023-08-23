import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import numpy as np
import torch.cuda
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

from grounding.data_processing.datasets_train import get_train_and_val_dataloaders
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

MODEL_SAVE_FILENAME = 'checkpoint.pth.tar'

parser = argparse.ArgumentParser(description='Training of a conditioned language model.')

parser.add_argument('--name', type=str, help='Name of experiment')
parser.add_argument('--no_image', action='store_true')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run')
parser.add_argument('--eval_every', type=int, default=220, help='Nb of update steps between evaluations')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--debug', action='store_true')

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.debug:
        full_exp_name = 'debug'
    else:
        full_exp_name = f"EXP-{args.name}__lr_{args.lr}__bs_{args.batch_size}__eval_every__{args.eval_every}"
    result_path = Path('./results/') / full_exp_name
    log_path = Path('./logs/') / full_exp_name
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    with open(result_path/'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    return result_path, log_path


def announce_start_training(args: argparse.Namespace) -> None:
    print('##############################')
    print('Number of Epochs = ', args.epochs)
    print('Batch size = ', args.batch_size)
    print('Learning rate = ', args.lr)
    print('Device: ', device)
    print('Use_image: ', not args.no_image)
    if args.debug:
        print('DEBUG MODE')
    print('##############################')


def forward_pass(batch, model: ImageConditionedLLMOnDecoder):
    input_texts, input_image_feats, output_texts = batch

    input_tokenized: BatchEncoding = model.tokenizer(input_texts, padding='longest', return_tensors='pt')
    decoder_input_toks, decoder_input_att_mask, decoder_image_features, output_toks = model.prepare_decoder_input_output_data(
        input_image_feats, output_texts)

    output: Seq2SeqLMOutput = model.forward(
        input_token_ids=input_tokenized['input_ids'].to(device),
        input_att_mask=input_tokenized['attention_mask'].to(device),
        decoder_input_token_ids=decoder_input_toks.to(device),
        decoder_input_att_mask=decoder_input_att_mask.to(device),
        image_features=decoder_image_features.to(device),
        output_token_ids=output_toks.to(device)
    )
    loss = output.loss
    perplexity = torch.exp(loss)
    return loss, perplexity


def validate_model(dataloader: DataLoader,
                   model: ImageConditionedLLMOnDecoder,
                   writer: SummaryWriter,
                   global_idx: int,
                   split: str):
    model.eval()
    val_losses = []
    val_perplexities = []

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            loss, perplexity = forward_pass(batch, model)
            val_losses.append(loss.mean().data.item())
            val_perplexities.append(perplexity.mean().data.item())

    avg_loss = np.array(val_losses).mean()
    avg_perplexity = np.array(val_perplexities).mean()

    writer.add_scalar(f"Loss/{split}", avg_loss, global_idx)
    writer.add_scalar(f"Perplexity/{split}", avg_perplexity, global_idx)
    return avg_loss, avg_perplexity


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def launch_training(args: Namespace):
    model: ImageConditionedLLMOnDecoder = ImageConditionedLLMOnDecoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                            amsgrad=False)

    print("Loading data...")
    train_fraction = 0.05 if args.debug else 1.0
    train_dataloader, val_seen_dataloader = get_train_and_val_dataloaders(
        batch_size=args.batch_size,
        clasp_mode=False,
        num_workers=1,
        train_fraction=train_fraction
    )
    result_path, log_path = setup_logging(args)
    writer = SummaryWriter(str(log_path))
    announce_start_training(args)

    global_idx = 0
    val_seen_loss_best = np.inf
    for epoch in range(args.epochs):
        avg_train_losses = []
        train_perplexities = []
        grad_norms = []

        with tqdm(train_dataloader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for i_batch, batch in enumerate(tepoch):
                model.train()
                optimizer.zero_grad()

                loss, perplexity = forward_pass(batch, model)
                loss.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                grad_norm = get_grad_norm(model)
                optimizer.step()

                train_loss = loss.mean().data.item()
                train_perplexity = perplexity.mean().data.item()

                avg_train_losses.append(train_loss)
                train_perplexities.append(train_perplexity)
                grad_norms.append(grad_norm)

                writer.add_scalar('Loss/train', train_loss, global_idx)
                writer.add_scalar('Perplexity/train', train_perplexity, global_idx)
                writer.add_scalar('Grad_norm/train', grad_norm, global_idx)

                global_idx += 1

                if global_idx % args.eval_every == 0:
                    val_seen_loss, val_seen_pp = validate_model(val_seen_dataloader, model, writer, global_idx, 'val_seen')

                    tepoch.set_postfix({'Loss/val_seen': val_seen_loss,
                                            'Perplex/val_seen': val_seen_pp})
                    writer.add_scalar('Loss/val_seen', val_seen_loss, global_idx)
                    writer.add_scalar('Perplexity/val_seen', val_seen_pp, global_idx)

                    if val_seen_loss < val_seen_loss_best:
                        val_seen_loss_best = val_seen_loss
                        torch.save(
                            {
                                'global_idx': global_idx,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_seen_loss': val_seen_loss
                            },
                            result_path / MODEL_SAVE_FILENAME
                        )
                writer.flush()
        train_loss = np.array(avg_train_losses).mean()
        train_pp = np.array(train_perplexities).mean()
        print(f"Average train loss: {train_loss}")
        print(f"Average train perplexity: {train_pp}")
    writer.close()


if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    launch_training(args)



