from random import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Generator,Discriminator,weights_init
import argparse
import numpy as np
import torch.optim as opt
from torch.autograd import Variable
from dataset import load_data, normalize
from torchvision.utils import save_image


parser=argparse.ArgumentParser("CycleGAN")
parser.add_argument("--img_path1",default="/home/udaykanth/project/ADRL/DC_GAN/bitmojis")
parser.add_argument("--img_path2",default="/home/udaykanth/project/ADRL/VAE_notme/data/celeba/img_align_celeba/img_align_celeba")
parser.add_argument("--batchsize",default=64)
parser.add_argument("--epochs",default=10)


def main():
    args=parser.parse_args()
    device="cuda:1"
    device2="cuda:1"
    k=10
    model_G_X_Y=Generator().to(device)
    model_G_Y_X=Generator().to(device)
    model_D_X=Discriminator().to(device2)
    model_D_Y=Discriminator().to(device2)

    model_D_X.apply(weights_init)
    model_D_Y.apply(weights_init)

    model_G_X_Y.apply(weights_init)
    model_G_Y_X.apply(weights_init)

    # model_G_X_Y.load_state_dict(torch.load("./saving/gen1/gen_X_Y_5.pth",map_location=device))
    # model_G_Y_X.load_state_dict(torch.load("./saving/gen2/genY_X_5.pth",map_location=device))
    # model_D_X.load_state_dict(torch.load("./saving/D1/D_X_0.pth",map_location=device2))
    # model_D_Y.load_state_dict(torch.load("./saving/D2/D_Y_0.pth",map_location=device2))

    mse_loss=nn.MSELoss().to(device)
    mse_loss1=nn.MSELoss().to(device2)
    l1_loss=nn.L1Loss().to(device)

    G_opt=opt.Adam(list(model_G_X_Y.parameters()) + list(model_G_Y_X.parameters()),lr=0.0002, betas=(0.5,0.99))
    D_X_opt=opt.Adam(model_D_X.parameters(),lr=0.0002, betas=(0.5,0.99))
    D_Y_opt=opt.Adam(model_D_Y.parameters(),lr=0.0002, betas=(0.5,0.99))

    data=load_data(args.img_path1,args.img_path2)
    real_data=DataLoader(dataset=data,batch_size=args.batchsize,shuffle=True)

    gen_loss=[]
    dis_loss=[]
    for epoch in range(args.epochs):
        real_images=tqdm(real_data)
        model_G_X_Y.train()
        model_G_Y_X.train()
        model_D_X.train()
        model_D_Y.train()
        loss_g=[]
        loss_d=[]
        batch_id=0
        for imgs in real_images:
            batch_id+=1
            img_X,img_Y=imgs
            X_imgs=Variable(img_X).to(device)
            Y_imgs=Variable(img_Y).to(device)
            real=Variable(torch.FloatTensor(np.ones(X_imgs.shape[0])), requires_grad=False).to(device)
            fake=Variable(torch.FloatTensor(np.zeros(X_imgs.shape[0])), requires_grad=False).to(device)

            # training discriminator
            for _ in range(2):
                D_X_opt.zero_grad()
                fake_x=model_G_Y_X(Y_imgs)
                fake_x_out=model_D_X(fake_x)
                real_x_out=model_D_X(X_imgs)
                D_X_loss=mse_loss1(real_x_out,real)+mse_loss1(fake_x_out,fake)
                D_X_loss.backward()
                D_X_opt.step()

                D_Y_opt.zero_grad()
                real_y_out=model_D_Y(Y_imgs)
                fake_y=model_G_X_Y(X_imgs)
                fake_y_out=model_D_Y(fake_y)
                D_Y_loss=mse_loss1(real_y_out,real)+mse_loss1(fake_y_out,fake)
                D_Y_loss.backward()
                D_Y_opt.step()
                D_loss=(D_X_loss+D_Y_loss)/2




            # training generator
            G_opt.zero_grad()
            fake_y=model_G_X_Y(X_imgs)
            fake_x=model_G_Y_X(Y_imgs)

            fake_y_out=model_D_Y(fake_y)
            fake_x_out=model_D_X(fake_x)

            x_reconstructed=model_G_Y_X(fake_y)
            y_reconstructed=model_G_X_Y(fake_x)

            loss_G_X_Y=mse_loss(fake_y_out,real)
            loss_G_Y_X=mse_loss(fake_x_out,real)

            X_cycle=l1_loss(x_reconstructed,X_imgs)
            Y_cycle=l1_loss(y_reconstructed,Y_imgs)

            G_loss=loss_G_X_Y+loss_G_Y_X+k*(X_cycle+Y_cycle)

            G_loss.backward(retain_graph=True)
            G_opt.step()

            loss_g.append(G_loss.item())
            loss_d.append(D_loss.item())
            if(batch_id%100==0):
                save_image((model_G_X_Y(X_imgs).data[:64]), "outputs/genx_y%d_%d.png" % (batch_id,epoch), nrow=8, normalize=True)
                save_image((model_G_Y_X(Y_imgs).data[:64]), "outputs/geny_x%d_%d.png" % (batch_id,epoch), nrow=8, normalize=True)

            real_images.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D_X(G_Y_X(Y)): %.4f D_Y(G_X_Y(X)): %.4f' % (
                epoch,args.epochs,D_loss.item(),G_loss.item(),fake_x_out.mean(),fake_y_out.mean()))


        gen_loss.append(loss_g)
        dis_loss.append(loss_d)
        if(epoch % 1 ==0):
            torch.save(model_G_X_Y.state_dict(),"saving/gen1/gen_X_Y_{}.pth".format(epoch))
            torch.save(model_G_Y_X.state_dict(),"saving/gen2/gen_Y_X_{}.pth".format(epoch))
            torch.save(model_D_X.state_dict(),"saving/D1/D_X_{}.pth".format(epoch))
            torch.save(model_D_Y.state_dict(),"saving/D2/D_Y_{}.pth".format(epoch))



    np.savetxt("G_loss.csv",
           gen_loss,
           delimiter=", ",
           fmt='% s')
    np.savetxt("D_loss.csv",
           dis_loss,
           delimiter=", ",
           fmt='% s')

        
if __name__=="__main__":
    main()