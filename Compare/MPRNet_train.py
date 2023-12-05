import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
from tqdm import tqdm
import numpy as np
from numpy import mean
from torchvision import transforms

from dataloaderPro import seed_torch
import dataloaderPro as dataloaderPro
import dataloaderTest
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

from Net.MPRNet import *

from losses.CL import L1_Charbonnier_loss
from utils import to_psnr


#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
# writer = SummaryWriter("./logs")

def weights_init_normal(m):  # 初始化权重
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(config):
	seed_torch(1143)

	device = torch.device(config.device)
	dehaze_net = Model().to(device)
	dehaze_net.apply(weights_init_normal)

	train_dataset = dataloaderPro.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, ps=config.image_size, perc=10)
	val_dataset = dataloaderTest.dehazing_loader(config.test_orig_images_path,
											 config.test_hazy_images_path, mode="test", perc=0, ps=config.image_size)


	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	criterion = L1_Charbonnier_loss().to(device)
	# criterion2 = SSIM().to(device)
	# criterion3 = PerceptualLoss(nn.MSELoss().to(device))# .to(device)
	optimizer = torch.optim.AdamW(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	scheduler_cos = CosineAnnealingLR(optimizer, config.num_epochs, 1e-7)

	#初始化tensorboard
	# tb_writer = SummaryWriter(config.tb_dir)
	# # config.tb_dir为保存tensorboard文件的路径
	# # tag即标题
	# train_tags = ['Train-Loss', 'Train-PSNR']
	# val_tags = ['Val-Loss', 'Val-PSNR']

	dehaze_net.train()

	# model = torch.nn.DataParallel(dehaze_net)
	# dehaze_net = dehaze_net.to(device)
	# load pretrained

	if os.path.isfile(config.snapshots_folder + "/dehazer.pth"):
		ckpt = torch.load(config.snapshots_folder + "/dehazer.pth")
		start_epoch = ckpt["epoch"]
		dehaze_net.load_state_dict(ckpt["net"])
		optimizer.load_state_dict(ckpt['optim'])
		scheduler_cos.load_state_dict(ckpt['sche'])

		# ckpt_dis = torch.load(config.snapshots_folder + "Dis.pth")
		# start_epoch_dis = ckpt_dis["epoch"]
		# dis.load_state_dict(ckpt_dis["net"])
		# optimizer_dis.load_state_dict(ckpt_dis['optim'])
		# scheduler_dis.load_state_dict(ckpt_dis['sche'])

	else:
		start_epoch = 0

	for epoch in tqdm(range(start_epoch, config.num_epochs)):
		train_psnr_list = []
		for iteration, (img_orig, img_haze) in enumerate(train_loader):

			img_orig = img_orig.to(device)
			img_haze = img_haze.to(device)


			dehaze_net.train()
			clean_image = dehaze_net(img_haze)[0]
			optimizer.zero_grad()
			train_loss = criterion(clean_image, img_orig) #  + 0.04 * (1 - criterion2(clean_image, img_orig)) + 0.005 * criterion3(clean_image, img_orig)#  - 0.01 * fake.detach().mean()
			# writer.add_scalar("Train-Loss", train_loss.item(), epoch)
			train_loss.backward()
			optimizer.step()
			# optimizer_dis.step()

			train_psnr_list.extend(to_psnr(clean_image, img_orig))
			train_psnr = sum(train_psnr_list) / len(train_psnr_list)
			# writer.add_scalar("Train-PSNR", train_psnr, epoch)

			# tensorboard可视化
			# for tag, value in zip(train_tags, [train_loss.item(), train_psnr]):
			# 	tb_writer.add_scalars(tag, {'Train': value}, epoch)

			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)

			if ((iteration + 1) % config.display_iter) == 0:
				print('\rEpoch: {}, Iteration: {}, Train_loss: {:.6f}, Train_PSNR:{:.6f}'.format(epoch, iteration,
																							   train_loss.item(),
																							   train_psnr), end='')
				# print("Loss at iteration", iteration + 1, ":", train_loss.item())
			if ((iteration + 1) % config.snapshot_iter) == 0:
				torch.save(dehaze_net.state_dict(), config.snapshots_folder + "/Epoch" + str(epoch) + '.pth')
				# torch.save(dis.state_dict(), config.snapshots_folder + "Dis_Epoch" + str(epoch) + '.pth')

		save_dict = {}
		save_dict["net"] = dehaze_net.state_dict()
		save_dict["epoch"] = epoch
		save_dict['optim'] = optimizer.state_dict()
		save_dict['sche'] = scheduler_cos.state_dict()
		torch.save(save_dict, config.snapshots_folder + "/dehazer.pth")

		scheduler_cos.step()
		# scheduler_dis.step()
		# Validation Stage

	dehaze_net.eval()
	for epoch in tqdm(range(0, config.num_epochs)):
		val_psnr_list = []
		os.makedirs(config.sample_output_folder+'/{}_Epoch'.format(epoch), exist_ok=True)
		dehaze_net.load_state_dict(torch.load(config.snapshots_folder + "/Epoch" + str(epoch) + '.pth'))
		for iter_val, (img_orig, img_haze) in enumerate(val_loader):

			img_orig = img_orig.to(device)
			img_haze = img_haze.to(device)

			clean_image = dehaze_net(img_haze)[0]

			val_loss = criterion(clean_image, img_orig)
			# writer.add_scalar("Val-Loss", val_loss.item(), epoch)

			val_psnr_list.extend(to_psnr(clean_image, img_orig))
			val_psnr = sum(val_psnr_list) / len(val_psnr_list)
			# writer.add_scalar("Val-PSNR", val_psnr.item(), epoch)

			# for tag, value in zip(val_tags, [val_loss.item(), val_psnr]):
			# 	tb_writer.add_scalars(tag, {'Val': value}, epoch)

			print('\rEpoch:{}, Val_Loss:{}, Val_PSNR:{}'.format(epoch, val_loss.item(), val_psnr), end='')

			torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+'/{}_Epoch/'.format(epoch)+str(iter_val+1)+".jpg")

		os.rename(config.sample_output_folder+'/{}_Epoch'.format(epoch), config.sample_output_folder+'/{:.6f}_psnr_{}Epoch'.format(mean(val_psnr_list), epoch))
		thisPath = config.sample_output_folder+'/{:.6f}_psnr_{}Epoch'.format(mean(val_psnr_list), epoch)
		for img in os.listdir(thisPath):
			os.remove(os.path.join(thisPath, img))
		print('Epoch: {},Train_loss: {}, Val-Loss: {}, Val_PSNR: {}'.format(epoch, train_loss, val_loss.item(), val_psnr))


if __name__ == "__main__":
	abla = 'MPRNet'
	model_list = r"Model/{}".format(abla)
	sample_list = r'Val/{}'.format(abla)
	os.makedirs(model_list, exist_ok=True)
	os.makedirs(sample_list, exist_ok=True)
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--device', type=str, default='cuda:0')
	parser.add_argument('--orig_images_path', type=str,
						default=r"Data/Train/images/")
	parser.add_argument('--hazy_images_path', type=str,
						default=r"Data/Train/data/")
	parser.add_argument('--test_orig_images_path', type=str,
						default=r"Data/Val/images/")
	parser.add_argument('--test_hazy_images_path', type=str,
						default=r"Data/Val/data/")

	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--val_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=200)
	parser.add_argument('--snapshots_folder', type=str, default=model_list)
	parser.add_argument('--sample_output_folder', type=str, default=sample_list)
	# parser.add_argument('--tb_dir', type=str, default="logs/")
	parser.add_argument('--image_size', type=int, default=128)


	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)

	train(config)








	
