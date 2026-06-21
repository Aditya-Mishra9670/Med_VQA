import argparse
import os
import sys
from pathlib import Path

MMBERT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MMBERT_DIR))

from path_config import configure_huggingface_cache, default_checkpoint_dir, default_converted_data_dir, default_hf_cache_dir, default_output_dir, select_device

configure_huggingface_cache(default_hf_cache_dir())

from utils import seed_everything, Model, VQAMed, train_one_epoch, validate, test, load_data, LabelSmoothing, train_img_only, val_img_only, test_img_only, make_grad_scaler
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
import warnings
import albumentations as A
import pretrainedmodels
from albumentations.core.composition import OneOf
from albumentations.pytorch.transforms import ToTensorV2

warnings.simplefilter("ignore", UserWarning)


def primary_val_score(val_acc):
    if isinstance(val_acc, dict):
        return val_acc.get('val_total_acc', val_acc.get('total_acc', 0))
    return val_acc


def log_wandb(metrics):
    if wandb.run is not None:
        wandb.log(metrics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Finetune on ImageClef 2019")

    parser.add_argument('--run_name', type = str, required = False, default = "model", help = "run name for wandb")
    parser.add_argument('--data_dir', '--converted_data_dir', dest='data_dir', type = str, required = False, default = str(default_converted_data_dir()), help = "path for converted data")
    parser.add_argument('--model_dir', type = str, required = False, default = None, help = "path to load pretrained weights")
    parser.add_argument('--save_dir', '--checkpoint_dir', dest='save_dir', type = str, required = False, default = str(default_checkpoint_dir()), help = "path to save weights")
    parser.add_argument('--output_dir', type = str, required = False, default = str(default_output_dir()), help = "path to save generated outputs")
    parser.add_argument('--cache_dir', type = str, required = False, default = None, help = "cache directory")
    parser.add_argument('--category', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")
    parser.add_argument('--wandb', action = 'store_true', default = False, help = "enable Weights & Biases logging")
    parser.add_argument('--allow_directml', action = 'store_true', default = False, help = "allow DirectML as a local fallback when CUDA is unavailable")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default = 16, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-4, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0.3, help = "hidden dropout probability")
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 312, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--num_vis', type = int, required = True, help = "num of visual embeddings")

    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.save_dir = str(Path(args.save_dir).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    configure_huggingface_cache(default_hf_cache_dir(cache_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        wandb.init(project='medvqa', name = args.run_name, config = args)

    seed_everything(args.seed)

    train_df, val_df, test_df = load_data(args)

    if args.category:
            
        train_df = train_df[train_df['category']==args.category].reset_index(drop=True)
        val_df = val_df[val_df['category']==args.category].reset_index(drop=True)
        test_df = test_df[test_df['category']==args.category].reset_index(drop=True)

        train_df = train_df[~train_df['answer'].isin(['yes', 'no'])].reset_index(drop = True)
        val_df = val_df[~val_df['answer'].isin(['yes', 'no'])].reset_index(drop = True)
        test_df = test_df[~test_df['answer'].isin(['yes', 'no'])].reset_index(drop = True)

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    val_df = df[df['mode']=='val'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes



    device = select_device(args.allow_directml)
    print(f"Using device: {device}")

    model = Model(args)

    if args.use_pretrained:
        if not args.model_dir:
            raise ValueError("--model_dir is required when --use_pretrained is set")
        model.load_state_dict(torch.load(args.model_dir))


    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)


        
    model.to(device)

    if args.wandb:
        wandb.watch(model, log='all')


    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor)


    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = make_grad_scaler(device, args.mixed_precision)


    train_tfm = transforms.Compose([
                                    
                                    transforms.ToPILImage(),
                                    transforms.RandomResizedCrop(224,scale=(0.75,1.25),ratio=(0.75,1.25)),
                                    transforms.RandomRotation(10),
                                    # Cutout(),
                                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    val_tfm = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_tfm = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




    traindataset = VQAMed(train_df, imgsize = args.image_size, tfm = train_tfm, args = args)
    valdataset = VQAMed(val_df, imgsize = args.image_size, tfm = val_tfm, args = args)
    testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args)

    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    best_acc1 = 0
    best_acc2 = 0
    best_loss = np.inf
    counter = 0

    for epoch in range(args.epochs):

        print(f'Epoch {epoch+1}/{args.epochs}')


        train_loss, _, _, _, _ = train_one_epoch(trainloader, model, optimizer, criterion, device, scaler, args, idx2ans)
        val_loss, predictions, val_acc, val_bleu = validate(valloader, model, criterion, device, scaler, args, val_df,idx2ans)
        test_loss, predictions, acc, bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
        val_score = primary_val_score(val_acc)
        scheduler.step(val_loss)
     

        if not args.category:

            log_dict = acc
            
            for k,v in bleu.items():
                log_dict[k] = v

            log_dict['train_loss'] = train_loss
            log_dict['test_loss'] = test_loss
            log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]

            log_wandb(log_dict)

        else:

            log_wandb({'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'learning_rate': optimizer.param_groups[0]["lr"],
                        f'val_{args.category}_acc': val_acc,
                        f'val_{args.category}_bleu': val_bleu,
                        f'{args.category}_acc': acc,
                        f'{args.category}_bleu': bleu}) 



        if not args.category:
            # print("val_acc =", val_acc)
            # print("type =", type(val_acc))
            if val_score > best_acc1:
                torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}_acc.pt'))
                best_acc1 = val_score

        else:

            if val_score > best_acc1:
                print('Saving model')
                torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}_acc.pt'))
                best_acc1 = val_score


        if val_score > best_acc2:
            counter = 0
            best_acc2 = val_score
        else:
            counter+=1
            if counter > 20:
                break      
