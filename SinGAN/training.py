import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(opt,Gs,Zs,reals,NoiseAmp):
    '''

    :param opt: config parameters
    :param Gs: []
    :param Zs: [[Gaussian noise [1, 3, 36, 36]], ]
    :param reals: []
    :param NoiseAmp: []
    :return:
    '''
    real_ = functions.read_image(opt) ## read image, map image to (-1, 1), convert image to pytorch tensor [N, C, W, H]
    in_s = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt) ## resize image based on opt.scale1 (0.213) -> [1, 3, 256, 256]
    reals = functions.creat_reals_pyramid(real,reals,opt) # realis a image pyramid list [(26), (33), (42), (55), (71), (92) ... , (256)] len = 10

    nfc_prev = 0

    while scale_num<opt.stop_scale+1: # opt.stop_scale + 1 = 9 + 1 = 10
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        # opt.nfc_init = 32, @ scale_num = 0, opt.nfc = 32 @ scale_num = 1, opt.nfc = 32
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        # opt.min_nfc_init = 32, @ scale_num = 0, opt.min_nfc_init = 32, @ scale_num = 1, opt.min_nfc = 32 ## only used in train_paint

        opt.out_ = functions.generate_dir2save(opt) # TrainedModels/path_01/scale_factor=0.750000,alpha=10
        opt.outf = '%s/%d' % (opt.out_,scale_num) # TrainedModels/path_01/scale_factor=0.750000,alpha=10/0

        try:
            os.makedirs(opt.outf) ## Generate a folder to save the image, and trained network
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)

        if (nfc_prev==opt.nfc): # @ scale_num = 0 (nfc_prev = 0 != opt.nfc = 32)
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,Gs,Zs,in_s,NoiseAmp,opt)
        # @ scale 0 z_curr is a Gaussian generated noise [1,3,36,36], in_s is all zero [1,3,26,26], G_curr is current generator net
        # @ scale 1, z_curr is all zeros [1, 3, 43, 43], in_s is all zero [1, 3, 26, 26]

        G_curr = functions.reset_grads(G_curr,False) ## Change all the requires_grads flage to be False. Aviod data copy for evaluations
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp) #[1]

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc # 32
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,centers=None):
    '''

    :param netD: currD
    :param netG: currG
    :param reals: a list of image pyramid
    :param Gs: list of prev netG
    :param Zs: [[Gaussian noise [1, 3, 36, 36]], ]
    :param in_s: 0-> all zero [1, 3, 26, 26]
    :param NoiseAmp: [] -> [1]
    :param opt: config
    :param centers:
    :return:
    '''

    real = reals[len(Gs)] # find the current level image xn
    opt.nzx = real.shape[2] #+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3] #+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2) # 5
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2) # 5


    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0

    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha # 10

    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy]) ## Notice that the gererated noise has 3 channels [3, 26, 26]
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device) ## z_opt is now all zero [1, 3, 26, 26]
    z_opt = m_noise(z_opt) ## z_opt is now [1, 3, 36, 36] -> [1, 3, 43, 43]

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter): # niter = 2000
        if (Gs == []) & (opt.mode != 'SR_train'): ## SR stands for super resolution
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy]) # [1, 1, 26, 26] ## Generated Gaussian Noise
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy)) # [1, 3, 36, 36]
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy]) # [1, 1, 26, 26]
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy)) # [1, 3, 36, 36]
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy])
            noise_ = m_noise(noise_) # scale = 1, noise_ [1, 3, 43, 43]

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps): ## Dsteps = 3
            # train with real
            netD.zero_grad()
            output = netD(real).to(opt.device) # real [1, 3, 26, 26] -> output [1, 1, 16, 16]
            #D_real_map = output.detach()
            errD_real = -output.mean() #-a W-GAN loss
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0): # first iteration training in this level
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev # full of 0 [1, 3, 26, 26]
                    prev = m_image(prev) #[1, 3, 36, 36]
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev) #[1, 3, 36, 36]
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    ## given a new noise, prev is a image generated by previous Generator with bilinear upsampling [1, 3, 33, 33]
                    prev = m_image(prev) ## [1, 3, 43, 43]

                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt) ## [1, 3, 33, 33]
                    ## z_prev is a image generated using a specific set of z group, for the purpose of reconstruction loss
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev) ## [1, 3, 43, 43]
            else:
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                ## Sample another image generated by the previous generator
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp * noise_+ prev ## [1, 3, 43, 43] new noise is equal to the prev generated image plus the gaussian noise.

            fake = netG(noise.detach(), prev)
            # detach() make sure that the gradients don't go to the noise.
            # noise:[1, 3, 36, 36] -> [1, 3, 43, 43]
            # prev:[1, 3, 36, 36] -> [1, 3, 43, 43] first step prev = 0, second step prev = a image generated by previous Generator with bilinaer upsampling

            # fake shape [1, 3, 26, 26] -> [1, 3, 33, 33]
            output = netD(fake.detach()) # output shape [1, 1, 16, 16] -> [1, 1, 23, 23]
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach()) ## errD for each iteration

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps): ## Gsteps = 3
            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean() # Generator want to make output as large as possible.
            errG.backward(retain_graph=True)
            if alpha!=0: ## alpha = 10
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp * z_opt + z_prev
                ## for the first scale z_prev = 0, z_opt = gausian noise, opt.noise_amp = 1 [1, 3, 36, 36]
                ## for the second scale z_prev image generated by Gn, z_opt is all zeros [1, 3, 43, 43]
                rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev), real)
                rec_loss.backward(retain_graph = True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss) ## ErrG for each iteration
        D_real2plot.append(D_x) ##  discriminator loss on real
        D_fake2plot.append(D_G_z) ## discriminator loss on fake
        z_opt2plot.append(rec_loss) ## reconstruction loss

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt) ## save netG, netD, z_opt, opt is used to parser output path
    return z_opt,in_s,netG
    # first scale z_opt: generated Gaussion noise [1, 3, 36, 36], in_s = all 0 [1, 3, 26, 26] netG: generator
    # second scale z_opt: all zeros [1, 3, 43, 43], in_s = [1, 3, 26, 26], all zeros

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    '''

    :param Gs: [G0]
    :param Zs: [[Gaussion Noise (1, 3, 36, 36)]]
    :param reals: [image pyramid]
    :param NoiseAmp: [1]
    :param in_s: all zeros [1, 3, 26, 26]
    :param mode: 'rand'
    :param m_noise:
    :param m_image:
    :param opt:
    :return:
    '''
    G_z = in_s #[1, 3, 26, 26] all zeros, image input for the corest level
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                ## z [1, 3, 26, 26]
                z = m_noise(z) ## z [1, 3, 36, 36]
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]] ## G_z [1, 3, 26, 26]
                G_z = m_image(G_z) ## G_z [1, 3, 36, 36] all zeros
                z_in = noise_amp*z+G_z ## [1, 3, 36, 36] Gaussian noise
                G_z = G(z_in.detach(),G_z) ## [1, 3, 26, 26] output of previous generator
                G_z = imresize(G_z,1/opt.scale_factor,opt) ## output upsampling (bilinear) [1, 3, 34, 34]
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]] ## resize the image to be compatible with current G [1, 3, 33, 33]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]] ## [1, 3, 26, 26] all zeros
                G_z = m_image(G_z) ## [1, 3, 36, 36] all zeros
                z_in = noise_amp*Z_opt+G_z  ## [1, 3, 36, 36] @ scale 1, it's scale 0's fixed gaussian
                G_z = G(z_in.detach(),G_z) ## [1, 3, 26, 26]
                G_z = imresize(G_z,1/opt.scale_factor,opt) ## [1, 3, 34, 34]
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]  ## [1, 3, 33, 33]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):
    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
