import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.distributed as dist
import numpy as np

class DINOLoss(nn.Module):
    def __init__(self, start_teacher_temp, end_teacher_temp,
                n_crops, n_epochs, warmup_epochs,
                student_temp=0.1, center_momentum=0.9):
        self.student_temp = student_temp
        self.n_crops = n_crops
        self.center_momentum = center_momentum
        self.teacher_temp_schedule = np.concatenate(
            np.linspace(start_teacher_temp,
                        end_teacher_temp,
                        warmup_epochs),
            np.ones(n_epochs - warmup_epochs) * end_teacher_temp
        )

    def forward(self, student_output, teacher_output, epoch):
        student_out =  student_output  / self.student_temp 
        student_out = student_out.chunk(self.n_crops)   

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, term_count = 0
        for index, chunk in enumerate(teacher_out):
            for view in range(len(student_out)):
                if view == index:
                    continue
                loss = torch.sum(-chunk * F.log_softmax(student_out[view], dim=-1), dim=-1)
                total_loss += loss.mean()
                term_count += 1
        total_loss /= term_count
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = torch.sum(teacher_out, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_out) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

