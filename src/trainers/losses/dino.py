"""DINOLoss module implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DINOLoss(nn.Module):
    """Implementation of the DINO v1 loss function."""

    def __init__(
        self,
        out_dim: int,
        base_teacher_temp: float,
        final_teacher_temp: float,
        n_crops: int,
        n_epochs: int,
        warmup_epochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        """Initialize DINOLoss module.

        Args:
            out_dim (int): Output dimension of the network.
            base_teacher_temp (float): Initial temperature for the teacher.
            final_teacher_temp (float): Final temperature for the teacher.
            n_crops (int): Number of crops used during training.
            n_epochs (int): Total number of training epochs.
            warmup_epochs (int): Number of warmup epochs for the teacher temperature.
            student_temp (float, optional): Temperature for the student. Defaults to 0.1
            center_momentum (float, optional): Momentum for the center update. Defaults to 0.9.
        """
        super().__init__()
        self.student_temp = student_temp
        self.n_crops = n_crops
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(base_teacher_temp, final_teacher_temp, warmup_epochs),
                np.ones(n_epochs - warmup_epochs) * final_teacher_temp,
            )
        )

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, epoch: int) -> torch.Tensor:
        """Compute DINO loss.

        Args:
            student_output (torch.Tensor): Output from the student network.
            teacher_output (torch.Tensor): Output from the teacher network.
            epoch (int): Current training epoch.

        Returns:
            torch.Tensor: Computed DINO loss.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        clamped_epoch = max(0, min(epoch, len(self.teacher_temp_schedule) - 1))
        temp = self.teacher_temp_schedule[clamped_epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, term_count = 0, 0
        for index, chunk in enumerate(teacher_out):
            for view in range(len(student_out)):
                if view == index:
                    # We skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-chunk * F.log_softmax(student_out[view], dim=-1), dim=-1)
                total_loss += loss.mean()
                term_count += 1
        total_loss /= term_count
        self._update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor) -> None:
        """Update the center used for teacher output normalization.

        Args:
            teacher_out (torch.Tensor): Output from the teacher network.
        """
        batch_center = torch.sum(teacher_out, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_out)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
