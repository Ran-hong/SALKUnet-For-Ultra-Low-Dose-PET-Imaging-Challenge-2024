from runner.runner import Runner
import os
import time
import torch
from torch.nn import DataParallel
import torch.nn as nn
import torch.nn.functional as F
from datasets.datasetCreator import datasetFactory
from models.modelCreator import modelFactory
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import restore_checkpoint, save_checkpoint
from metrics import compute_psnr
import nibabel as nib


class Runner_e2e(Runner):
    def __init__(self, config, workdir):
        self.config = config 
        self.workdir = workdir
    def train(self):
        flag = time.time()
        # create floder
        tb_dir = os.path.join(self.workdir, "tensorboard")
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        writer = SummaryWriter(tb_dir)
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create model
        model = modelFactory(self.config)
        model = DataParallel(model, device_ids=[0,1])
        model.to(self.device)
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr = self.config.optim.lr,
            betas = self.config.optim.betas,
            eps = self.config.optim.eps,
            weight_decay = self.config.optim.weight_decay
        )
        state = dict(optimizer=optimizer, model=model, epoch=0)
        # 创建ckpt meta，用来继续中断的训练
        checkpoint_dir = os.path.join(self.workdir, "checkpoints")
        checkpoint_meta_dir = os.path.join(self.workdir, "checkpoint_meta", "checkpoint.pth")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
            os.makedirs(os.path.dirname(checkpoint_meta_dir))
        state = restore_checkpoint(checkpoint_meta_dir, state, self.device)
        initial_epoch = int(state["epoch"])

        loss_fn = nn.MSELoss()
        # create dataset
        dataset = datasetFactory(self.config)
        dataloader = DataLoader(dataset, 
                                batch_size=self.config.train.batch_size,
                                shuffle=True,
                                num_workers=8,)
        
        print("Init completed ! cost time: {:.3f}".format(time.time() - flag))
        for epoch in range(initial_epoch, self.config.train.num_train_epoch):
            loss_total = psnr_total = 0.
            for step, batch in enumerate(dataloader):
                loss_batch = psnr_batch = 0.
                lq = batch['lq'].to(self.device)
                hq = batch['hq'].to(self.device)
                pred = model(lq)
                loss = loss_fn(pred, hq)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch = loss.item()
                psnr_batch = compute_psnr(pred, hq)
                print("\r{} [epoch:{}/{} batch:{}/{}] [loss_batch:{:.4f}] [psnr_batch:{:.4f}]".format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    epoch, self.config.train.num_train_epoch, step, len(dataloader),
                    loss_batch, psnr_batch,
                ))
                iterations = epoch * len(dataloader) + step
                writer.add_scalar(f"{tb_dir}/train_loss", loss_batch, iterations)
                writer.add_scalar(f"{tb_dir}/train_psnr", psnr_batch, iterations)
                loss_total += loss_batch
                psnr_total += psnr_batch
            print("\n{} [epoch:{}] [loss:{:.4f}] [psnr:{:.4f}]".format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                batch, loss_total/len(dataloader), psnr_total/len(dataloader),
            ))
            writer.add_scalar(f"{tb_dir}/avg_loss", loss_total/len(dataloader), epoch)
            writer.add_scalar(f"{tb_dir}/avg_psnr", psnr_total/len(dataloader), epoch)

            if epoch != 0 and epoch % self.config.train.snapshot_freq == 0:
                save_epoch = epoch // self.config.train.snapshot_freq
                save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{save_epoch}.pth"), state)
                print("save checkpoint completed")

            if epoch != 0 and epoch % self.config.train.snapshot_freq_for_preemption == 0:
                save_checkpoint(checkpoint_meta_dir, state)
                print("save checkpoint for preemeption completed")

        writer.close()
        pass

    @torch.no_grad()
    def eval(self, eval_dir, eval_pth):
        # create floder
        tb_dir = os.path.join(eval_dir, "tensorboard")
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        writer = SummaryWriter(tb_dir)
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = modelFactory(self.config)
        model = DataParallel(model, device_ids=[0,1])
        model.to(self.device)
        optimizer = torch.optim.Adam(params = model.parameters(),lr = self.config.optim.lr)
        state = dict(optimizer=optimizer, model=model, epoch=0)
        state = restore_checkpoint(eval_pth, state, self.device)
        initial_epoch = int(state["epoch"])
        print(f"restore complete, eval start")
        dataset = datasetFactory(self.config)
        dataloader = DataLoader(dataset, 
                                batch_size=self.config.train.batch_size,
                                shuffle=False,
                                num_workers=8,)
        
        psnr_total = 0.
        for step, batch in enumerate(dataloader):
            psnr_batch = 0.
            lq = batch['lq'].to(self.device)
            hq = batch['hq'].to(self.device)
            pred = model(lq)
            psnr_batch = compute_psnr(pred, hq)
            print("\r [batch: {}/{}] [psnr_batch: {:.4f}]".format(
                step, len(dataloader), psnr_batch,
            ))
            writer.add_scalar(f"{tb_dir}/eval_psnr", psnr_batch, step)
            psnr_total += psnr_batch
            '''
            if (step == 1):
                import pickle
                fr = open("block", 'wb')
                pickle.dump(pred, fr)
                fr.close()
            '''
        print('\n Eval Complete, psnr_avg: {:.4f}'.format(psnr_total / len(dataloader)))


    @torch.no_grad()
    def test(self, test_dir, test_pth):
        # create floder
        res_dir = os.path.join(test_dir, "results")
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = modelFactory(self.config)
        model = DataParallel(model, device_ids=[0, 1])
        model.eval()
        model.to(self.device)
        optimizer = torch.optim.Adam(params = model.parameters(),lr = self.config.optim.lr)
        state = dict(optimizer=optimizer, model=model, epoch=0)
        state = restore_checkpoint(test_pth, state, self.device)
        print(f"restore complete, test start")
        dataset = datasetFactory(self.config)
        dataloader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle=False,
            num_workers=0,
        )
        for step, batch in enumerate(dataloader):
            upper = batch["upper"].to(self.device)
            pred = torch.zeros_like(upper).to(self.device)
            if upper.shape[2] == 673:
                parts = [model(upper[:, :, i*112:i*112+112, 52:-52, 52:-52]) for i in range(6)]
                for i, part in enumerate(parts):
                    pred[:, :, i*112:i*112+112, 52:-52, 52:-52] = part
            elif upper.shape[2] == 644:
                parts = [model(upper[:, :, i*64:i*64+64, 76:-76, 76:-76]) for i in range(10)]
                for i, part in enumerate(parts):
                    pred[:, :, i*64:i*64+64, 76:-76, 76:-76] = part

            pred = pred.detach().squeeze(0).squeeze(0).cpu()


            pred = ((pred * batch["dose"]) / (batch["weight"] * 1000)).permute(1, 2, 0).unsqueeze(-1).cpu().numpy()
            nii_img = nib.Nifti1Image(pred, batch["affine"].squeeze(0))
            save_path = f"{res_dir}/{batch['dir'][0]}/{batch['name'][0]}"
            if not os.path.exists(f"{res_dir}/{batch['dir'][0]}"):
                os.mkdir(f"{res_dir}/{batch['dir'][0]}")
            nib.save(nii_img, save_path)
            print("save one")
        pass