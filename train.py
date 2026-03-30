from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augment import PostProcessConfig, RandomPostProcessPerturbation
from data import JsonlImageDataset, build_path_substitutions
from model import ConvNeXtForgeryClassifier


class SwanLabLogger:
    def __init__(self, run: Any | None = None) -> None:
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(data, step=step)

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen multi-scale ConvNeXt classifier for AI image source identification."
    )
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "train.csv")
    parser.add_argument("--val-manifest", type=Path, default=ROOT / "test.csv")
    parser.add_argument("--class-map", type=Path, default=ROOT / "class_map.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="convnext_tiny.dinov3_lvd1689m",
    )
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--align-dim", type=int, default=192)
    parser.add_argument("--local-dim", type=int, default=128)
    parser.add_argument("--global-dim", type=int, default=128)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--postprocess-prob", type=float, default=0.5)
    parser.add_argument("--postprocess-max-ops", type=int, default=2)
    parser.add_argument("--eval-postprocess-prob", type=float, default=0.5)
    parser.add_argument("--eval-postprocess-max-ops", type=int, default=2)
    parser.add_argument("--jpeg-quality-min", type=int, default=35)
    parser.add_argument("--jpeg-quality-max", type=int, default=95)
    parser.add_argument("--webp-quality-min", type=int, default=35)
    parser.add_argument("--webp-quality-max", type=int, default=95)
    parser.add_argument("--blur-radius-max", type=float, default=1.6)
    parser.add_argument("--resize-scale-min", type=float, default=0.5)
    parser.add_argument("--noise-std-max", type=float, default=4.0)
    parser.add_argument("--crop-scale-min", type=float, default=0.75)
    parser.add_argument("--sharpen-factor-max", type=float, default=2.0)
    parser.add_argument("--brightness-delta", type=float, default=0.2)
    parser.add_argument("--contrast-delta", type=float, default=0.2)
    parser.add_argument("--saturation-delta", type=float, default=0.2)
    parser.add_argument("--gamma-delta", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--path-substitution",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Rewrite prefixes inside manifest paths without editing the manifest files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="AI-Identification")
    parser.add_argument("--swanlab-workspace", type=str, default=None)
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-description", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default="cloud")
    parser.add_argument("--swanlab-logdir", type=Path, default=None)
    parser.add_argument("--swanlab-group", type=str, default=None)
    parser.add_argument("--swanlab-public", action="store_true")
    parser.add_argument("--swanlab-tags", action="append", default=[])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.lower()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = mps_backend is not None and mps_backend.is_available()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device, but torch.cuda.is_available() is False. "
            "This usually means the current Python environment does not have a working CUDA-enabled PyTorch build, "
            "or the process cannot see the NVIDIA driver/GPU."
        )

    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested with --device mps, but torch.backends.mps.is_available() is False."
        )

    return torch.device(device_arg)


def log_device_info(device: torch.device) -> None:
    print(
        "Runtime device:"
        f" requested={device}"
        f" cuda_available={torch.cuda.is_available()}"
        f" cuda_device_count={torch.cuda.device_count()}"
        f" cuda_version={torch.version.cuda}"
    )
    if device.type == "cuda":
        print(f"Using CUDA device {device.index or 0}: {torch.cuda.get_device_name(device)}")


def load_class_map(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: sanitize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_config_value(item) for item in value]
    return value


def init_swanlab_logger(args: argparse.Namespace, num_classes: int) -> SwanLabLogger:
    if not args.use_swanlab:
        return SwanLabLogger()

    try:
        import swanlab
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SwanLab logging is enabled but the 'swanlab' package is not installed. "
            "Install it with `python3 -m pip install swanlab`."
        ) from error

    logdir = args.swanlab_logdir or (args.output_dir / "swanlog")
    run = swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.swanlab_experiment_name,
        description=args.swanlab_description,
        job_type="eval" if args.eval_only else "train",
        config=sanitize_config_value({
            **vars(args),
            "num_classes": num_classes,
        }),
        logdir=str(logdir),
        mode=args.swanlab_mode,
        group=args.swanlab_group,
        tags=args.swanlab_tags or None,
        public=args.swanlab_public,
    )
    return SwanLabLogger(run)


def build_model(args: argparse.Namespace, num_classes: int) -> ConvNeXtForgeryClassifier:
    return ConvNeXtForgeryClassifier(
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        pretrained=not args.no_pretrained,
        backbone_checkpoint=args.backbone_checkpoint,
        align_dim=args.align_dim,
        local_dim=args.local_dim,
        global_dim=args.global_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    )


def build_transforms(
    model: ConvNeXtForgeryClassifier, image_size: int, args: argparse.Namespace
) -> tuple[Any, Any]:
    data_config = resolve_data_config({}, model=model.backbone)
    data_config["input_size"] = (3, image_size, image_size)
    base_train_transform = create_transform(
        **data_config,
        is_training=True,
        auto_augment=None,
        re_prob=0.0,
    )
    eval_transform = create_transform(
        **data_config,
        is_training=False,
    )
    train_postprocess = RandomPostProcessPerturbation(
        PostProcessConfig(
            probability=args.postprocess_prob,
            max_ops=args.postprocess_max_ops,
            jpeg_quality_min=args.jpeg_quality_min,
            jpeg_quality_max=args.jpeg_quality_max,
            webp_quality_min=args.webp_quality_min,
            webp_quality_max=args.webp_quality_max,
            blur_radius_max=args.blur_radius_max,
            resize_scale_min=args.resize_scale_min,
            noise_std_max=args.noise_std_max,
            crop_scale_min=args.crop_scale_min,
            sharpen_factor_max=args.sharpen_factor_max,
            brightness_delta=args.brightness_delta,
            contrast_delta=args.contrast_delta,
            saturation_delta=args.saturation_delta,
            gamma_delta=args.gamma_delta,
        )
    )
    eval_postprocess = RandomPostProcessPerturbation(
        PostProcessConfig(
            probability=args.eval_postprocess_prob,
            max_ops=args.eval_postprocess_max_ops,
            jpeg_quality_min=args.jpeg_quality_min,
            jpeg_quality_max=args.jpeg_quality_max,
            webp_quality_min=args.webp_quality_min,
            webp_quality_max=args.webp_quality_max,
            blur_radius_max=args.blur_radius_max,
            resize_scale_min=args.resize_scale_min,
            noise_std_max=args.noise_std_max,
            crop_scale_min=args.crop_scale_min,
            sharpen_factor_max=args.sharpen_factor_max,
            brightness_delta=args.brightness_delta,
            contrast_delta=args.contrast_delta,
            saturation_delta=args.saturation_delta,
            gamma_delta=args.gamma_delta,
        )
    )
    train_transform = transforms.Compose([train_postprocess, base_train_transform])
    eval_transform = transforms.Compose([eval_postprocess, eval_transform])
    return train_transform, eval_transform


def build_dataloaders(
    args: argparse.Namespace,
    train_transform: Any,
    eval_transform: Any,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    substitutions = build_path_substitutions(args.path_substitution)
    train_dataset = JsonlImageDataset(
        manifest_path=args.train_manifest,
        transform=train_transform,
        data_root=args.data_root,
        path_substitutions=substitutions,
    )
    val_dataset = JsonlImageDataset(
        manifest_path=args.val_manifest,
        transform=eval_transform,
        data_root=args.data_root,
        path_substitutions=substitutions,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    results = []
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / targets.size(0)))
    return results


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(state: dict[str, Any], output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / filename)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_top1", 0.0))


def move_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, labels


def progress_write(message: str) -> None:
    tqdm.write(message)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    logger: SwanLabLogger,
    global_step: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    autocast_enabled = args.amp and device.type == "cuda"
    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"train {epoch}",
        dynamic_ncols=True,
        leave=False,
    )
    for step, batch in enumerate(progress_bar, start=1):
        global_step += 1
        images, labels = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(images)
            loss = criterion(outputs["logits"], labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()

        top1, top5 = accuracy(outputs["logits"], labels, topk=(1, 5))
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top1.item() * batch_size / 100.0
        total_top5 += top5.item() * batch_size / 100.0
        total_samples += batch_size
        avg_loss = total_loss / total_samples
        avg_top1 = 100.0 * total_top1 / total_samples
        avg_top5 = 100.0 * total_top5 / total_samples
        lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            top1=f"{avg_top1:.2f}",
            top5=f"{avg_top5:.2f}",
            lr=f"{lr:.6f}",
        )

        if step % args.print_freq == 0 or step == len(loader):
            logger.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_top1": top1.item(),
                    "train/batch_top5": top5.item(),
                    "train/running_loss": avg_loss,
                    "train/running_top1": avg_top1,
                    "train/running_top5": avg_top5,
                    "train/lr": lr,
                    "train/epoch": epoch,
                },
                step=global_step,
            )
    progress_bar.close()
    progress_write(
        f"[train] epoch={epoch} loss={avg_loss:.4f} top1={avg_top1:.2f} top5={avg_top5:.2f} lr={lr:.6f}"
    )

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_top1 / total_samples,
        "top5": 100.0 * total_top5 / total_samples,
        "global_step": global_step,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=split,
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress_bar:
        images, labels = move_to_device(batch, device)
        outputs = model(images)
        loss = criterion(outputs["logits"], labels)
        top1, top5 = accuracy(outputs["logits"], labels, topk=(1, 5))

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top1.item() * batch_size / 100.0
        total_top5 += top5.item() * batch_size / 100.0
        total_samples += batch_size
        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            top1=f"{100.0 * total_top1 / total_samples:.2f}",
            top5=f"{100.0 * total_top5 / total_samples:.2f}",
        )

    metrics = {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_top1 / total_samples,
        "top5": 100.0 * total_top5 / total_samples,
    }
    progress_bar.close()
    progress_write(
        f"[{split}] loss={metrics['loss']:.4f} top1={metrics['top1']:.2f} top5={metrics['top5']:.2f}"
    )
    return metrics


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    log_device_info(device)
    class_map = load_class_map(args.class_map)
    num_classes = int(class_map["num_classes"])
    logger = init_swanlab_logger(args, num_classes)

    try:
        model = build_model(args, num_classes=num_classes)
        train_transform, eval_transform = build_transforms(model, args.image_size, args)
        train_loader, val_loader = build_dataloaders(args, train_transform, eval_transform, device)

        model = model.to(device)
        total_params, trainable_params = count_parameters(model)
        print(
            f"Model built with total_params={total_params:,} trainable_params={trainable_params:,} "
            f"backbone={args.backbone_name}"
        )
        logger.log(
            {
                "meta/total_params": total_params,
                "meta/trainable_params": trainable_params,
            },
            step=0,
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        )
        scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

        start_epoch = 1
        best_top1 = 0.0
        global_step = 0
        if args.resume is not None:
            last_epoch, best_top1 = load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch = last_epoch + 1
            global_step = last_epoch * len(train_loader)
            print(f"Resumed from {args.resume} at epoch {last_epoch} with best_top1={best_top1:.2f}")

        if args.eval_only:
            val_metrics = evaluate(model, val_loader, criterion, device, split="val")
            logger.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/top1": val_metrics["top1"],
                    "val/top5": val_metrics["top5"],
                    "val/best_top1": best_top1,
                },
                step=start_epoch - 1,
            )
            return

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                args=args,
                logger=logger,
                global_step=global_step,
            )
            global_step = int(train_metrics.pop("global_step"))
            val_metrics = evaluate(model, val_loader, criterion, device, split="val")

            logger.log(
                {
                    "train/epoch_loss": train_metrics["loss"],
                    "train/epoch_top1": train_metrics["top1"],
                    "train/epoch_top5": train_metrics["top5"],
                    "val/loss": val_metrics["loss"],
                    "val/top1": val_metrics["top1"],
                    "val/top5": val_metrics["top5"],
                    "val/best_top1": max(best_top1, val_metrics["top1"]),
                },
                step=epoch,
            )

            checkpoint = {
                "epoch": epoch,
                "best_top1": best_top1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            save_checkpoint(checkpoint, args.output_dir, "latest.pt")

            if val_metrics["top1"] >= best_top1:
                best_top1 = val_metrics["top1"]
                checkpoint["best_top1"] = best_top1
                save_checkpoint(checkpoint, args.output_dir, "best.pt")
            progress_write(f"Epoch {epoch} complete. best_top1={best_top1:.2f}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
