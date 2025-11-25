"""Pre-training module for Vision-Transformer."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import VisionTransformer, VisionTransformerWithPretrainingHeads

from .base import BaseTrainer
from .losses import DINOLoss


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
            student_model (VisionTransformer | VisionTransformerWithPretrainingHeads): The model to be trained.
            teacher_model (VisionTransformer | VisionTransformerWithPretrainingHeads): The teacher model.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): Optimizer type.
            scheduler_class (str): Learning rate scheduler type.
            **kwargs (dict): Additional arguments for the base trainer.

        Returns:
            Trainer: An instance of the Trainer class.
        """
        super().__init__(student_model, teacher_model, learning_rate, optimizer_class, scheduler_class, **kwargs)

        # initialize dino loss
        self._dino_loss = DINOLoss(
            out_dim=kwargs.get("dino_out_dim"),
            base_teacher_temp=kwargs.get("dino_base_teacher_temp"),
            final_teacher_temp=kwargs.get("dino_final_teacher_temp"),
            n_crops=2 + kwargs.get("num_local_crops"),
            n_epochs=kwargs.get("num_epochs"),
            warmup_epochs=kwargs.get("dino_warmup_epochs"),
        ).to(next(self.student_model.parameters()).device)

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

    def evaluate(self, val_loader: DataLoader, epoch: int, device: torch.device) -> dict[str, float]:
        """Evaluate the ViT model.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.
            device (torch.device): Device to perform evaluation on.

        Returns:
            dict[str, float]: A dictionary of validation losses.
        """
        self.student_model.eval()
        self.teacher_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = [im.to(device) for im in batch]  # List of 2 global views + num_local_crops of [B, C, H, W]
                # run forward pass for teacher and student
                teacher_outputs = self.teacher_model(images[:2])
                student_outputs = self.student_model(images)
                loss = self._dino_loss(student_outputs, teacher_outputs, epoch, update_teacher=False)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

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

        # training loop
        for epoch in range(num_epochs):
            total_epoch_train_loss = 0.0
            total_epoch_val_loss = 0.0
            num_val_runs = 0
            epoch_step = 0
            for batch in train_loader:
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

                # move batch to device
                images = [im.to(device) for im in batch]  # List of 2 global views + num_local_crops of [B, C, H, W]
                # run forward pass for teacher and student
                teacher_outputs = self.teacher_model(images[:2])
                student_outputs = self.student_model(images)
                loss = self._dino_loss(student_outputs, teacher_outputs, epoch, update_teacher=True)

                # record loss
                log_dict["train_loss"] = loss.item()

                # backward pass
                loss.backward()
                total_epoch_train_loss += loss.item()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)

                # for the first n epochs, cancel gradients for the last layer in the DinoHead
                if epoch == 0:  # TODO: make this a parameter
                    for name, p in self.student_model.named_parameters():
                        if "last_layer" in name:
                            p.grad = None

                # optimizer step
                self.optimizer.step()

                # step weight decay scheduler
                self.wd_scheduler.step()

                # step lr scheduler
                self.lr_scheduler.step()

                # update the teacher model
                with torch.no_grad():
                    m = self.momentum_scheduler.step()
                    for param_q, param_k in zip(self.student_model.parameters(), self.teacher_model.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.data)

                # log metrics
                if global_step % log_interval_steps == 0:
                    # run validation if provided
                    val_metrics = None
                    if val_loader is not None:
                        val_metrics = self.evaluate(val_loader, epoch, device)
                        if val_metrics is not None:
                            num_val_runs += 1
                            log_dict["val_loss"] = val_metrics["loss"]
                            total_epoch_val_loss += val_metrics["loss"]

                    log_str = json.dumps(
                        {
                            "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                            "step": global_step,
                            "train_loss": loss.item(),
                            "val_loss": val_metrics["loss"] if val_metrics is not None else None,
                            "lr": self.optimizer.param_groups[0]["lr"],
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
            avg_epoch_train_loss = total_epoch_train_loss / epoch_step
            log_dict_epoch = {
                "epoch": epoch + 1,
                "step": global_step,
                "avg_train_loss": avg_epoch_train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if val_loader is not None and num_val_runs > 0:
                avg_epoch_val_loss = total_epoch_val_loss / num_val_runs
                log_dict_epoch["avg_val_loss"] = avg_epoch_val_loss

            log_str = json.dumps(log_dict_epoch)
            progress_bar.write(log_str)
            if self.wandb_writer is not None:
                self.write_losses_to_wandb(global_step, log_dict_epoch)

            # save best model checkpoint based on the specified metric
            if save_best and best_loss is not None:
                # determine which loss metric to use
                if loss_metric_for_best_model == "train":
                    current_loss = avg_epoch_train_loss
                elif loss_metric_for_best_model == "val" and num_val_runs > 0:
                    current_loss = avg_epoch_val_loss
                else:
                    raise ValueError("Invalid loss_metric_for_best_model.")

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

        # close wandb writer
        if self.wandb_writer is not None:
            self.wandb_writer.finish()
