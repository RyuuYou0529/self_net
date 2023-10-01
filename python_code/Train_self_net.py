import os
import torch
import Self_net_architecture
import argparse
import torch.optim as optim
import pytorch_dataset
import checkpoint
import itertools
from ssim import SSIM
from ssim_3d import SSIM3D
from Charbonnier_loss import L1_Charbonnier_loss
import time

from torch.utils.tensorboard import SummaryWriter

def adjust_lr(init_lr,optimizer,epoch,step,gamma):
    if (epoch+1)%step==0:
        times=(epoch+1)/step
        lr=init_lr*gamma**times
        for params in optimizer.param_groups:
            params['lr']=lr


def backward_D_basic(lambda_gan,netD, criterionGAN,real, fake,type,device):
    """Calculate GAN loss for the discriminator
           Parameters:
               netD (network)      -- the discriminator D
               real (tensor array) -- real images
               fake (tensor array) -- images generated by a generator
           Return the discriminator loss.
           We also call loss_D.backward() to calculate the gradients.
           """
    # Real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake =criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    if type=='lsgan' or type=='vanilla':
        loss_D = lambda_gan*(loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    else:
        print('warning in calculating loss_D')


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def train(train_set,netD_A,netD_B,netG_A,netG_B,deblur_net,args,device,criterionGAN,criterionL1,SSIM_loss,optimD,optimG,optim_deblur):

    lambda_cycle = 1
    lambda_deblur=1
    lambda_feedback=0
    lambda_gan=0.1
    lambda_ssim=0.1


    netD_A.train()
    netD_B.train()

    netG_A.train()
    netG_B.train()
    deblur_net.train()

    fakeA_pool=pytorch_dataset.ImagePool(50)
    fakeB_pool=pytorch_dataset.ImagePool(50)

    Loss_D=0.0
    Loss_G_GAN=0.0
    Loss_G_cycle=0.0

    Loss_feedback=0.0
    Loss_deblur=0.0

    writer = SummaryWriter(log_dir=os.path.join(args.path, 'checkpoint/tensorboard'))
    total_iters= 0
    for epoch in range(args.epochs):
        param1 = optimD.param_groups[0]
        param2=optimG.param_groups[0]
        adjust_lr(args.learning_rate_D, optimD, epoch, 15, 0.5)
        adjust_lr(args.learning_rate_G, optimG, epoch, 15, 0.5)
        adjust_lr(args.learning_rate_G, optim_deblur, epoch, 15, 0.5)

        print(f'[epoch:{epoch+1}]  learning_rate_D:{param1["lr"]}')
        print(f'[epoch:{epoch+1}]  learning_rate_G:{param2["lr"]}')


        if epoch+1>=2:
            lambda_feedback=0.1


        for i, data in enumerate(train_set):
            batch_time_start = time.time()
            total_iters += args.batch_size

            #degradation modeling
            A=data['hr_deg']
            B=data['lr']
            C=data['hr']

            A=A.to(device)
            B=B.to(device)
            C=C.to(device)

            fake_B=netG_A(A)      #syn_lr

            fake_A=netG_B(B)

            set_requires_grad(deblur_net,False)
            fake_C1 = deblur_net(fake_B)

            rec_A=netG_B(fake_B)

            rec_B=netG_A(fake_A)

            #optimize G
            set_requires_grad(netD_A, False)
            set_requires_grad(netD_B, False)

            optimG.zero_grad()

            # GAN loss D_A(G_A(A))
            loss_G_A = lambda_gan*criterionGAN(netD_A(fake_B), True)
            writer.add_scalar('Loss_G/loss_G_A', loss_G_A, global_step=total_iters)
            # GAN loss D_B(G_B(B))
            loss_G_B = lambda_gan*criterionGAN(netD_B(fake_A), True)
            writer.add_scalar('Loss_G/loss_G_B', loss_G_B, global_step=total_iters)


            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = lambda_cycle*criterionL1(rec_A,A)+lambda_ssim*(1-SSIM_loss(rec_A,A))
            writer.add_scalar('Loss_G/loss_cycle_A', loss_cycle_A, global_step=total_iters)

            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = lambda_cycle*criterionL1(rec_B,B)+lambda_ssim*(1-SSIM_loss(rec_B,B))
            writer.add_scalar('Loss_G/loss_cycle_B', loss_cycle_B, global_step=total_iters)


            loss_feedback = lambda_feedback*(criterionL1(fake_C1, C)+lambda_ssim*(1-SSIM_loss(fake_C1,C)))
            writer.add_scalar('Loss_G/loss_feedback', loss_feedback, global_step=total_iters)

            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_feedback
            writer.add_scalar('Loss_G/loss_G', loss_G, global_step=total_iters)

            loss_G.backward()
            optimG.step()

            #optimize D
            if (i+1)%2==0:
                set_requires_grad(netD_A, True)
                set_requires_grad(netD_B, True)

                optimD.zero_grad()

                fake_B = fakeB_pool.query(fake_B)
                loss_D_A = backward_D_basic(lambda_gan,netD_A, criterionGAN, B, fake_B, 'lsgan', device)

                fake_A = fakeA_pool.query(fake_A)
                loss_D_B = backward_D_basic(lambda_gan,netD_B, criterionGAN, A, fake_A, 'lsgan', device)

                optimD.step()

            else:
                loss_D_A=torch.Tensor([0.0])
                loss_D_B=torch.Tensor([0.0])
            writer.add_scalar('Loss_D/loss_D_A', loss_D_A, global_step=total_iters)
            writer.add_scalar('Loss_D/loss_D_B', loss_D_B, global_step=total_iters)

            #Checkpoint
            Loss_D += loss_D_A.item()+loss_D_B.item()
            Loss_G_GAN += loss_G_A.item()+loss_G_B.item()
            Loss_feedback += loss_feedback.item()
            Loss_G_cycle += loss_cycle_A.item()+loss_cycle_B.item()

            if (i + 1) % args.log_interval == 0:
                print(f'[epochs: {epoch+1} {(i+1)*args.batch_size}/{len(train_set.dataset)}] [degradation modeling stage]: '
                      f'loss_D:{Loss_D/(args.log_interval*lambda_gan):4f}, '
                      f'loss_G: {Loss_G_GAN/(args.log_interval*lambda_gan):4f}, '
                      f'loss_G_cycle:{Loss_G_cycle / (args.log_interval*lambda_cycle):4f}, '
                      f'loss_feedback:{Loss_feedback/(args.log_interval):4f}')
                Loss_D = 0.0
                Loss_G_GAN = 0.0
                Loss_G_cycle = 0.0
                Loss_feedback=0.0

            #deblurring
            set_requires_grad(deblur_net, True)
            fake_B = netG_A(A)  # syn_hr_deg
            fake_C2 = deblur_net(fake_B.detach())

            optim_deblur.zero_grad()

            loss_deblur = lambda_deblur * criterionL1(fake_C2, C) + lambda_ssim * (1 - SSIM_loss(fake_C2, C))
            writer.add_scalar('Loss_Deblur/loss_deblur', loss_deblur, global_step=total_iters)

            loss_deblur.backward()
            optim_deblur.step()

            # Checkpoint
            Loss_deblur += loss_deblur.item()

            if (i + 1) % args.log_interval == 0:
                print(f'[epochs: {epoch+1} {(i+1)*args.batch_size}/{len(train_set.dataset)}]  [deblurring stage]: loss_deblur:{Loss_deblur /(args.log_interval*lambda_deblur):4f}')
                Loss_deblur = 0.0

            if (i + 1) % args.imshow_interval == 0:
                checkpoint.intermediate_output(deblur_net, device, args.path, args.normalize_mode, epoch, (i+1)*args.batch_size)
                torch.save(deblur_net.state_dict(),os.path.join(args.path, 'checkpoint/saved_models/', 'deblur_net/', f'{epoch + 1}_{(i+1)*args.batch_size}.pkl'))
                torch.save(netG_A.state_dict(),os.path.join(args.path, 'checkpoint/saved_models/', 'netG_A/', f'{epoch + 1}_{(i+1)*args.batch_size}.pkl'))
                torch.save(netG_B.state_dict(),os.path.join(args.path, 'checkpoint/saved_models/', 'netG_B/', f'{epoch + 1}_{(i+1)*args.batch_size}.pkl'))
                torch.save(netD_A.state_dict(),os.path.join(args.path, 'checkpoint/saved_models/', 'netD_A/', f'{epoch + 1}_{(i+1)*args.batch_size}.pkl'))
                torch.save(netD_B.state_dict(),os.path.join(args.path, 'checkpoint/saved_models/', 'netD_B/', f'{epoch + 1}_{(i+1)*args.batch_size}.pkl'))
            
            if (i + 1) % args.log_interval == 0:
                batch_time_end = time.time()
                print(f'[time]: {batch_time_end-batch_time_start :.6f} \n')

def main():
    parser=argparse.ArgumentParser(description='Self_Net')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--path', default='/home/ryuuyou/Project/self_net/data/visor/')

    parser.add_argument('--normalize_mode', type=str, default='min_max')
    parser.add_argument('--max_v', type=int, default=1)

    parser.add_argument('--learning_rate_G',type=float,default=1e-4)
    parser.add_argument('--learning_rate_D', type=float, default=1e-4)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_interval',type=int,default=5)
    parser.add_argument('--imshow_interval',type=int,default=250)

    parser.add_argument('--net_G',type=str,default='care')

    args=parser.parse_args()

    checkpoint_path=os.path.join(args.path, 'checkpoint/')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    model_saved_path=os.path.join(checkpoint_path, 'saved_models/')
    intermediate_results_path=os.path.join(checkpoint_path, 'intermediate_results/')

    if not os.path.exists(model_saved_path):
        os.mkdir(model_saved_path)

    # ==================
    def gene_sub_model_saved_path(sub_path: str):
        path = os.path.join(model_saved_path, sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
    gene_sub_model_saved_path('deblur_net')
    gene_sub_model_saved_path('netG_A')
    gene_sub_model_saved_path('netG_B')
    gene_sub_model_saved_path('netD_A')
    gene_sub_model_saved_path('netD_B')
    # ==================

    if not os.path.exists(intermediate_results_path):
        os.mkdir(intermediate_results_path)



    device=torch.device('cuda:0')

    input_nc=1
    output_nc=1

    netD_A = Self_net_architecture.define_D(input_nc=input_nc, ndf=64, netD='n_layers', n_layers_D=2, device=device,norm='instance')
    netD_B = Self_net_architecture.define_D(input_nc=input_nc, ndf=64, netD='n_layers', n_layers_D=2, device=device,norm='instance')

    netG_A = Self_net_architecture.define_G(input_nc=input_nc, output_nc=output_nc, ngf=32, netG=args.net_G, device=device,use_dropout=False,norm='instance')
    netG_B = Self_net_architecture.define_G(input_nc=input_nc, output_nc=output_nc, ngf=32, netG=args.net_G, device=device,use_dropout=False,norm='instance')

    deblur_net=Self_net_architecture.define_G(input_nc=input_nc, output_nc=output_nc, ngf=32, netG=args.net_G, device=device,use_dropout=False,norm='instance')

    criterionGAN=Self_net_architecture.GANLoss(gan_mode='lsgan').to(device)

    # SSIM_loss = SSIM(data_range=1, size_average=True, win_size=11, win_sigma=1.5, channel=1)
    SSIM_loss = SSIM3D(window_size=11)

    criterionL1=L1_Charbonnier_loss()
    optimG = optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()),lr=args.learning_rate_G, betas=(args.beta1, args.beta2))
    optimD=optim.Adam(itertools.chain(netD_A.parameters(),netD_B.parameters()),lr=args.learning_rate_D, betas=(args.beta1, args.beta2))
    optim_deblur=optim.Adam(deblur_net.parameters(),lr=args.learning_rate_G, betas=(args.beta1, args.beta2))

    train_set=pytorch_dataset.create_train_data(os.path.join(args.path, 'train_data/'),args.batch_size, args.normalize_mode)
    train(train_set,netD_A,netD_B,netG_A,netG_B,deblur_net,args,device,criterionGAN,criterionL1,SSIM_loss,optimD,optimG,optim_deblur)
    print('done training!')

import sys

if __name__=='__main__':
    f = open('train.log', 'w')
    sys.stdout = f
    sys.stderr = f

    main()

    f.close()
