"""Main entry point for Vision Transformer pre-training."""

import random

import numpy as np
import torch

import dataset
import models
import trainers
import utils

if __name__ == "__main__":
    # set up argument parser and parse args
    args = utils.parse_pretrain_args()
    print(args)

    # set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create train and validation dataloaders
    train_loader, val_loader = dataset.create_pretrain_dataloaders(
        val_split=args.val_split,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_local_crops=args.num_local_crops,
        local_crop_size=args.local_crop_size,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        seed=args.seed,
    )
    print(f"Loaded train dataloader with {len(train_loader.dataset)} samples.")
    if val_loader is not None:
        print(f"Loaded val dataloader with {len(val_loader.dataset)} samples.")

    # initialize models
    student_model = models.init_model(args, device, cls="pretraining")
    teacher_model = models.init_model(args, device, cls="pretraining")
    # copy student weights to teacher and freeze teacher parameters
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.requires_grad = False
    # print model summary
    total_params = sum(p.numel() for p in student_model.parameters())
    print(student_model)
    print(
        f"Instantiated ViT with:\n\ttotal #params: {total_params / 10**6:.2f}M "
        f"\n\tencoder #params: {sum(p.numel() for p in student_model.encoder.parameters()) / 10**6:.2f}M"
    )

    # initialize trainer
    trainer = trainers.init_trainer(student_model, teacher_model, train_loader, args, cls="pretraining")

    # train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        upload_model_to_hub=args.upload_model_to_hub,
        repo_id=args.repo_id,
        log_interval_steps=args.log_interval_steps,
        save_interval_steps=args.save_interval_steps,
        save_latest=args.save_latest,
        save_best=args.save_best,
    )
