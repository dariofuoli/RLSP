import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from parameters import params
from rlsp import RLSP
import numpy as np


class System:
    def __init__(self):

        # create checkpoint folder and initialize monitoring
        os.makedirs("logs/checkpoints", exist_ok=True)
        self.monitor = SummaryWriter("logs/tensorboard")

        # initialize model
        self.rlsp = RLSP().to(params["device"])

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.rlsp.parameters(), lr=params["lr"])

        # set iteration
        self.iter = 0

        # load last checkpoint if available
        if os.path.exists("logs/checkpoints/last.txt"):
            self.load_checkpoint()


    def train_step(self, x, y):

        self.optimizer.zero_grad()
        loss = F.mse_loss(self.rlsp(x), y)
        loss.backward()
        self.optimizer.step()

        self.iter += 1

        # monitor loss
        self.monitor.add_scalar("Training Loss", loss.item(), self.iter)
    
    def validation(self, loader_val):
        
        print("Checking convergence...")

        device = params["device"]

        loss = 0
        loss_psnr = 0
        for i, (x, y) in enumerate(loader_val.dataloader):
            
            print("Sequence: ", i)

            x = x.to(device)
            y = y.to(device)
       
            with torch.no_grad():
                rlsp_x = self.rlsp(x)    

                # clip and round to 8bit RGB       
                rlsp_x = torch.clamp(rlsp_x, 0, 255)
                rlsp_x = torch.round(rlsp_x)

                l2_loss = F.mse_loss(rlsp_x, y)
                loss += l2_loss
                loss_psnr += 10*torch.log10(255**2/l2_loss)

        loss = loss/loader_val.dataset.videos
        loss_psnr = loss_psnr/loader_val.dataset.videos

        self.monitor.add_scalar("Validation Loss", loss.item(), self.iter)
        self.monitor.add_scalar("Validation PSNR", loss_psnr.item(), self.iter)

    def save_checkpoint(self, name=None):

        print("Saving checkpoint...")
        if name is None:
            save_string = "logs/checkpoints/" + str(self.iter).zfill(7) + ".pt"
        else:
            save_string = "logs/checkpoints/" + name + "_" + str(self.iter).zfill(7) + ".pt"

        torch.save({
            "rlsp": self.rlsp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iter}, save_string)

        # set "last checkpoint" file
        with open("logs/checkpoints/last.txt", "w") as f:
            f.write(save_string)

        print("Saved checkpoint at: ", save_string)

    def load_checkpoint(self, file_name="last"):
        
        device = params["device"]

        if file_name == "last":
            print("Loading last checkpoint...")
            with open("logs/checkpoints/last.txt", "r") as f:
                path = f.readline().strip()
        else:
            print("Loading checkpoint: ", "logs/checkpoints/" + file_name)
            path = "logs/checkpoints/" + file_name

        if str(device) == "cpu":
            print(device)

            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)

        self.rlsp.load_state_dict(checkpoint["rlsp"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter = checkpoint["iteration"]

        print("Loaded checkpoint: ", path)

    def set_learningrate(self, lr):
        for p in self.optimizer.param_groups:
            p["lr"] = lr
            print("New learning rate: ", p["lr"] )
