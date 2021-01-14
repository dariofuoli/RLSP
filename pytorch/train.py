from parameters import params
from system import System
from dataloader_train import Loader as Loader_train
from dataloader_val import Loader as Loader_val
import time


device = params["device"]
print("***************** device: ", device, " / system time: ", time.ctime(), "*****************")

# initialize trainer and loader objects
system = System()
loader_train = Loader_train()
loader_val = Loader_val()

# start training
start = time.time()
for x, y in loader_train.dataloader:

    x = x.to(device)
    y = y.to(device)

    system.train_step(x=x, y=y)
    if system.iter % 100 == 0:
        print(int(system.iter/loader_train.epoch_size), system.iter, (time.time() - start)/3600)

    if system.iter % params["validation interval"] == 0:
        system.validation(loader_val=loader_val)

    if system.iter % params["save interval"] == 0:
        system.save_checkpoint()
