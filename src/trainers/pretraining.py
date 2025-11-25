"""Pre-training module for Vision-Transformer."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models import VisionTransformer, VisionTransformerWithPretrainingHeads

from .base import BaseTrainer
from .losses.dino_loss import DINOLoss


class PreTrainer(BaseTrainer):
    """Trainer class for self-supervised pre-training."""

    def __init__(
        self,
        student_model: VisionTransformer | VisionTransformerWithPretrainingHeads,
        teacher_model: VisionTransformer | VisionTransformerWithPretrainingHeads,
        learning_rate: float,
        optimizer_class: str,
        scheduler_class: str,
        **kwargs: dict,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model (VisionTransformer | VisionTransformerWithPretrainingHeads): The model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): Optimizer type.
            scheduler_class (str): Learning rate scheduler type.
            **kwargs (dict): Additional arguments for the base trainer.

        Returns:
            Trainer: An instance of the Trainer class.
        """
        super().__init__(student_model, teacher_model, learning_rate, optimizer_class, scheduler_class, **kwargs)

    def _get_model_attrs(self, model: nn.Module) -> dict:
        """Get model attributes for logging.

        Args:
            model (nn.Module): The model.

        Returns:
            dict: A dictionary of model attributes.
        """
        attrs = model.__dict__.copy()
        attrs.update({"model_type": type(model).__name__})
        attrs.update({"model_class": model.__class__.__name__})
        attrs.update({"model_module": model.__class__.__module__})
        attrs.update({"total_params": sum(p.numel() for p in model.parameters())})
        return attrs

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, **kwargs: dict) -> dict[str, float]:
        """Validate the ViT model.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            kwargs (dict): Additional arguments.

        Returns:
            dict[str, float]: A dictionary of validation losses.
        """
        pass

    def train(  # noqa: C901
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_epochs: int,
        save_dir: str = "./saved_models",
        use_wandb: bool = False,
        wandb_entity: str = None,
        wandb_project: str = None,
        wandb_name: str = None,
        upload_model_to_hub: bool = False,
        repo_id: str = None,
        log_interval_steps: int = 100,
        save_interval_steps: int = 200,
        save_latest: bool = False,
        save_best: bool = True,
        loss_metric_for_best_model: str = "train",
        **kwargs: dict,
    ) -> None:
        """Train the ViT model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader | None): DataLoader for validation data.
            num_epochs (int): Number of training epochs.
            save_dir (str): Directory to save model checkpoints.
            use_wandb (bool): Whether to use Weights & Biases for logging.
            wandb_entity (str): Weights & Biases entity name.
            wandb_project (str): Weights & Biases project name.
            wandb_name (str): Weights & Biases run name.
            upload_model_to_hub (bool): Whether to upload the model to Hugging Face Hub
            repo_id (str): Hugging Face Hub repository ID.
            log_interval_steps (int): Steps interval for logging training loss.
            save_interval_steps (int): Steps interval for saving model checkpoints.
            save_latest (bool): If True, overwrite the latest checkpoint instead of saving per save steps.
            save_best (bool): If True, track and save the best model checkpoint based on the specified loss metric.
            loss_metric_for_best_model (str): Metric to use for best model tracking ('train' or 'val').
            kwargs (dict): Additional arguments.

        Returns:
            None
        """
        # initialize logging (e.g., Weights & Biases)
        if use_wandb:
            self.init_wandb(
                entity=wandb_entity,
                project=wandb_project,
                name=wandb_name,
                config=self._get_model_attrs(self.student_model),
            )

        # Set up HuggingFace Hub Configuration
        if upload_model_to_hub:
            self.init_hf_api()

        # Set up tracking variables
        global_step = 0
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        progress_bar = tqdm(total=total_steps)
        best_loss = float("inf") if save_best else None

        # reset model to training mode
        self.student_model.train()
        self.student_model.zero_grad()
        torch.cuda.empty_cache()
        device = next(self.student_model.parameters()).device

        # preliminary checks
        if loss_metric_for_best_model == "val" and val_loader is None:
            raise ValueError("Cannot use 'val' metric for best model when val_loader is None")
        if upload_model_to_hub and repo_id is None:
            raise ValueError("repo_id must be specified when upload_model_to_hub is True")
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty.")
        if val_loader is not None and len(val_loader) == 0:
            raise ValueError("val_loader is empty.")
        if save_best and loss_metric_for_best_model not in ["train", "val"]:
            raise ValueError("loss_metric_for_best_model must be either 'train' or 'val'.")

        # initialize dino loss
        dino_loss = DINOLoss(
            out_dim=65536,
            start_teacher_temp=0.04,
            end_teacher_temp=0.04,
            n_crops=2 + 6,  # TODO: make this dynamic based on number of local and global views
            n_epochs=num_epochs,
            warmup_epochs=0,
        ).to(device)

        # training loop
        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            epoch_step = 0
            for batch in train_loader:
                # step weight decay scheduler
                if self.wd_schedule is not None:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        if i == 0:  # only the first group is regularized
                            param_group["weight_decay"] = self.wd_schedule[epoch_step]

                # track steps
                epoch_step += 1
                global_step += 1

                # zero gradients
                self.student_model.train()
                self.optimizer.zero_grad()

                # create log dict
                log_dict = {
                    "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                num_objectives = len(batch)
                total_batch_loss = 0.0

                # move batch to device
                images = [im.to(device) for im in batch]  # List of 2 global views + n local views of [B, C, H, W] each
                # run forward pass for teacher and student
                teacher_outputs = self.teacher_model(images[:2])
                student_outputs = self.student_model(images)
                loss = dino_loss(student_outputs, teacher_outputs, epoch)
                pass

                # TODO: Complete training loop

                # average loss across objectives
                loss = total_batch_loss / num_objectives
                log_dict["loss"] = loss.item()

                # backward pass
                loss.backward()
                total_epoch_loss += loss.item()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # optimizer step
                self.optimizer.step()

                # step the scheduler (note: we are stepping every batch)
                self.scheduler.step()

                # log metrics
                if global_step % log_interval_steps == 0:
                    log_str = json.dumps(
                        {
                            "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                            "step": global_step,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "loss": loss.item(),
                            **{f"loss_{k}": v for k, v in log_dict.items() if k.startswith("loss_") and k != "loss"},
                        }
                    )
                    progress_bar.write(log_str)
                    if self.wandb_writer is not None:
                        self.write_losses_to_wandb(global_step, log_dict)

                # update progress bar
                progress_bar.set_postfix(
                    {
                        "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                        "loss": f"{loss.item():.4f}",
                    }
                )
                progress_bar.update(1)

                # save model checkpoint
                if global_step % save_interval_steps == 0:
                    if not save_latest:
                        checkpoint_path = Path(save_dir) / f"step_{global_step}"
                    else:
                        checkpoint_path = Path(save_dir) / "latest"

                    self.save_pretrained(save_directory=checkpoint_path)
                    if upload_model_to_hub:
                        self.push_to_hub(
                            repo_id=repo_id,
                            commit_message=f"Training Step {global_step}",
                        )

            # end of epoch logging
            avg_epoch_loss = total_epoch_loss / epoch_step
            log_str = json.dumps(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss_avg": avg_epoch_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            progress_bar.write(log_str)
            if self.wandb_writer is not None:
                self.write_losses_to_wandb(global_step, {"loss_avg": avg_epoch_loss})

            # run validation
            if val_loader is not None:
                val_losses = self.validate(val_loader, **kwargs)
                log_str = json.dumps(
                    {"epoch": epoch + 1, "step": global_step, **{f"val_{k}": v for k, v in val_losses.items()}}
                )
                progress_bar.write(log_str)
                if self.wandb_writer is not None:
                    self.write_losses_to_wandb(
                        global_step,
                        {f"val_{k}": v for k, v in val_losses.items()},
                    )

            # save best model checkpoint based on the specified metric
            if save_best and best_loss is not None:
                # determine which loss metric to use
                if loss_metric_for_best_model == "train":
                    current_loss = avg_epoch_loss
                elif loss_metric_for_best_model == "val" and val_loader is not None:
                    current_loss = val_losses["loss_avg"]
                else:
                    raise ValueError("Invalid loss_metric_for_best_model or missing val_loader.")

                if current_loss < best_loss:
                    best_loss = current_loss
                    checkpoint_path = Path(save_dir) / "best_model"
                    self.save_pretrained(save_directory=checkpoint_path)
                    if upload_model_to_hub:
                        self.push_to_hub(
                            repo_id=repo_id,
                            commit_message=f"Best model at Step {global_step}",
                        )

        # close progress bar
        progress_bar.close()

        # final model save
        self.save_pretrained(save_directory=Path(save_dir) / "final_model")
        if upload_model_to_hub:
            self.push_to_hub(
                repo_id=repo_id,
                commit_message=f"Final model at Step {global_step}",
            )
