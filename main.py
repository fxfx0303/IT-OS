import json
from datasets.VidSTG import VidSTGDataset
import torch
import numpy as np
import random
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch import optim
from pathlib import Path
from models import build_model
# from model_new import build_model
from copy import deepcopy
from functools import partial
import utils.misc as utils_misc
import utils.dist as dist
from collections import namedtuple
import time, datetime
import argparse
from engine import evaluate, train_one_epoch
from models.postprocessors import build_postprocessors

# from models.vistr import build
# from datasets.ytvos import build as build_ytvos
# from datasets import build_dataset, get_coco_api_from_dataset
# from model_new.postprocessors import build_postprocessors
#
# import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument("--do_qa", action="store_true", help="Whether to do question answering")
    parser.add_argument(
        "--predict_final",
        action="store_true",
        help="If true, will predict if a given box is in the actual referred set. Useful for CLEVR-Ref+ only currently.",
    )
    parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")
    parser.add_argument(
        "--split_qa_heads", action="store_true", help="Whether to use a separate head per question type in vqa"
    )
    parser.add_argument(
        "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    )
    parser.add_argument(
        "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    )

    parser.add_argument("--coco_path", type=str, default="")
    parser.add_argument("--vg_img_path", type=str, default="")
    parser.add_argument("--vg_ann_path", type=str, default="")
    parser.add_argument("--clevr_img_path", type=str, default="")
    parser.add_argument("--clevr_ann_path", type=str, default="")
    parser.add_argument("--phrasecut_ann_path", type=str, default="")
    parser.add_argument(
        "--phrasecut_orig_ann_path",
        type=str,
        default="",
    )
    parser.add_argument("--modulated_lvis_ann_path", type=str, default="")

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr_drop", default=35, type=int)
    parser.add_argument(
        "--epoch_chunks",
        default=-1,
        type=int,
        help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    # parser.add_argument(
    #     "--text_encoder_type",
    #     default="roberta-base",
    #     choices=("roberta-base", "distilroberta-base", "roberta-large"),
    # )

    parser.add_argument(
        "--text_encoder_type",
        default="video_text/mdetr-main/transformer_pretrain",
        # choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument("--num_queries", default=360, type=int, help="Number of query slots")
    parser.add_argument("--num_queries_per_frame", default=10, type=int, help="Number of query slots per frame")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Segmentation
    parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--masks", action="store_true")

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss",
    )

    parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    parser.add_argument(
        "--no_contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_false",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # Loss coefficients
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--qa_loss_coef", default=1, type=float)
    parser.add_argument("--l1_loss_coef", default=0.1, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    # Run specific

    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="r101_mycode_temporal_test2_fuxian9", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=5, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # parser.add_argument('--output-dir', default='r101_mycode',
    #                         help='path where to save, empty for no saving')
    return parser

# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     parser.add_argument('--lr', default=1e-4, type=float)
#     parser.add_argument('--lr_backbone', default=1e-5, type=float)
#     parser.add_argument('--batch_size', default=1, type=int)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--epochs', default=18, type=int)
#     parser.add_argument('--lr_drop', default=12, type=int)
#     parser.add_argument('--clip_max_norm', default=0.1, type=float,
#                         help='gradient clipping max norm')
#
#     # Model parameters
#     parser.add_argument('--pretrained_weights', type=str, default="r101_pretrained.pth",
#                         help="Path to the pretrained model.")
#     # * Backbone
#     parser.add_argument('--backbone', default='resnet101', type=str,
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument('--dilation', action='store_true',
#                         help="If true, we replace stride with dilation in the last convolutional block (DC5)")
#     parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                         help="Type of positional embedding to use on top of the image features")
#
#     # * Transformer
#     parser.add_argument('--enc_layers', default=6, type=int,
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers', default=6, type=int,
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument('--dim_feedforward', default=2048, type=int,
#                         help="Intermediate size of the feedforward layers in the transformer blocks")
#     parser.add_argument('--hidden_dim', default=384, type=int,
#                         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument('--nheads', default=8, type=int,
#                         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_frames', default=36, type=int,
#                         help="Number of frames")
#     parser.add_argument('--num_queries', default=360, type=int,
#                         help="Number of query slots")
#     parser.add_argument('--pre_norm', action='store_true')
#
#     # * Segmentation
#     parser.add_argument('--masks', action='store_true',
#                         help="Train segmentation head if the flag is provided")
#
#     # Loss
#     parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
#                         help="Disables auxiliary decoding losses (loss at each layer)")
#     # * Matcher
#     parser.add_argument('--set_cost_class', default=1, type=float,
#                         help="Class coefficient in the matching cost")
#     parser.add_argument('--set_cost_bbox', default=5, type=float,
#                         help="L1 box coefficient in the matching cost")
#     parser.add_argument('--set_cost_giou', default=2, type=float,
#                         help="giou box coefficient in the matching cost")
#     # * Loss coefficients
#     parser.add_argument('--mask_loss_coef', default=1, type=float)
#     parser.add_argument('--dice_loss_coef', default=1, type=float)
#     parser.add_argument('--bbox_loss_coef', default=5, type=float)
#     parser.add_argument('--giou_loss_coef', default=2, type=float)
#     parser.add_argument('--eos_coef', default=0.1, type=float,
#                         help="Relative classification weight of the no-object class")
#
#     # dataset parameters
#     parser.add_argument('--dataset_file', default='ytvos')
#     parser.add_argument('--ytvos_path', type=str)
#     parser.add_argument('--remove_difficult', action='store_true')
#
#     parser.add_argument('--output_dir', default='r101_vistr',
#                         help='path where to save, empty for no saving')
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--seed', default=42, type=int)
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                         help='start epoch')
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--num_workers', default=4, type=int)
#
#     # distributed training parameters
#     parser.add_argument('--world_size', default=1, type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
#     return parser

def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    print(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.set_deterministic(True)
    #
    # ##修改
    # torch.set_deterministic(False)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    #创建模型
    model, criterion, contrastive_criterion, weight_dict = build_model(args)
    model.to(device)

    # Get a copy of the model for exponential moving averaged version of the model
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    #optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
            "lr": args.text_encoder_lr,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    #读入数据集
    train_dataset = VidSTGDataset("train", args)
    # train_dataset = build_ytvos("train", args)
    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
    # sampler_train = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler_train,
        collate_fn=partial(utils_misc.collate_fn, False),
        num_workers=args.num_workers,
    )

    if args.eval:
        val_type = 'test'
    else:
        val_type = 'test'

    val_dataset = VidSTGDataset(val_type, args)
    if args.distributed:
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    # sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    data_loader_val = DataLoader(
        val_dataset,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=partial(utils_misc.collate_fn, False),
        num_workers=args.num_workers,
    )

    if args.frozen_weights is not None:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model_ema" in checkpoint and checkpoint["model_ema"] is not None:
            model_without_ddp.detr.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint["model"], strict=False)

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # Used for loading weights from another model and starting a training from scratch. Especially useful if
    # loading into a model with different functionality.
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model_ema" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    if args.eval:
        test_stats = {}
        # test_model = model_ema if model_ema is not None else model
        test_model = model
###########test
        # evaluate函数
###############
        postprocessors = build_postprocessors(args, "VidSTG")
        print(f"Evaluating VidSTG")
        # evaluate(
        #     model=test_model,
        #     criterion=criterion,
        #     postprocessors=postprocessors,
        #     data_loader=data_loader_train,
        #     base_ds=None,
        #     device=device,
        #     output_dir='r101_vistr/',
        # )

        curr_test_stats = evaluate(
            model=test_model,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            postprocessors=postprocessors,
            weight_dict=weight_dict,
            data_loader=data_loader_val,
            device=device,
            args=args,
        )
        test_stats.update({"VidSTG_" + k: v for k, v in curr_test_stats.items()})
        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        print(log_stats)
        exit()

    #training函数
    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            data_loader=data_loader_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
        )

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            #每1轮保存一次
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        if epoch % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model
            postprocessors = build_postprocessors(args, "VidSTG")
            print(f"Evaluating VidSTG")
            curr_test_stats = evaluate(
                model=test_model,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=data_loader_val,
                device=device,
                args=args,
            )
            test_stats.update({"VidSTG_" + k: v for k, v in curr_test_stats.items()})
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.eval_skip == 0:
            if args.do_qa:
                metric = test_stats["gqa_accuracy_answer_total_unscaled"]
            else:
                metric = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_bbox" in k])

            if args.output_dir and metric > best_metric:
                best_metric = metric
                checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

