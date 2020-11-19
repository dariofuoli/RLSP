import torch


# set global parameters

params = {"lr": 10 ** -4,
          "bs": 1,
          "crop size h": 64,
          "crop size w": 64,
          "sequence length": 10,
          "validation sequence length": 20,
          "number of workers": 8,
          "layers": 7,
          "kernel size": 3,
          "filters": 128,
          "state dimension": 128,
          "factor": 4,
          "save interval": 50000,
          "validation interval": 10000, 
          "dataset root": "/dataset/",
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          }
