import argparse
import itertools
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from models import Generator_svhn, Discriminator, ResNet20
from tensorboardX import SummaryWriter
from data_loader import get_loader
import torch.nn.functional as F
from utils import compute_lambda_anneal, compute_gradient_penalty, _lr_factor,free_params, frozen_params, evaluate_losses

import math
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

def is_progress_interval(args, epoch):
    return epoch == args.epochs - 1 or (args.progress_intervals > 0 and epoch % args.progress_intervals == 0)

def train_dn(args, device):
    # =============================================================================
    experiment_path = args.experiment_path

    rate = args.latent_dim * math.log2(args.L)
    print(f"Rate: {rate}")
    # =============================================================================

    weight_path = os.path.join(args.model_save_dir, 'weights')
    try:
        os.makedirs(weight_path)
    except OSError:
        pass

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    latent_dim = args.latent_dim
    Lambda_s = 0.003
    print('Common randomness?', args.common)
    print('Working on latent_dim = {} when Lambda = {}, noisy level {}'.format(latent_dim, Lambda_s, args.noise_level))

    # Dataset
    dataloader, dataloader_test, unnormalizer = get_loader(args)
    test_set_size = len(dataloader_test.dataset)

    try:
        os.makedirs(os.path.join(args.model_save_dir, args.dataset))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(weight_path, args.dataset))
    except OSError:
        pass

    device = torch.device("cuda:0" if args.cuda else "cpu")

    # writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

    # create model
    generator = Generator_svhn(latent_dim, args.L, args.ql, args.stochastic, args.common).to(device)
    discriminator = Discriminator(input_channel=3).to(device)
    alpha1 = generator.encoder.alpha

    # =============================================================================
    # ---------------------------------------------------------------------
    # Define classifier model for SVHN 
    # ---------------------------------------------------------------------
    classifier1 = ResNet20().to(device)
    optimizer_classifier1 = optim.SGD(classifier1.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    cross_entropy_criterion = nn.CrossEntropyLoss()
    # =============================================================================

    criterion = torch.nn.MSELoss().to(device)

    # =============================================================================
    os.makedirs(f"{experiment_path}", exist_ok=True)

    with open(f"{experiment_path}/_perception_losses.csv", "w") as f:
        f.write("epoch,distortion_loss,perception_loss,rate,lambda\n")

    with open(f"{experiment_path}/_cross_entropy_losses.csv", "w") as f:
        f.write("epoch,distortion_loss,cross_entropy_loss,rate,lambda\n")

    with open(f"{experiment_path}/_accuracy.csv", "w") as f:
        f.write("epoch,distortion_loss,accuracy,rate,lambda\n")
    # =============================================================================

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_factor)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_factor)

    num_batches = len(dataloader)
    # Train GAN-Oral.

    n_cycles = 1 + args.n_critic

    disc_loss = torch.Tensor([-1])
    distortion_loss = torch.Tensor([-1])
    lambda_gp = 10

    # =============================================================================
    perception_loss = torch.Tensor([-1]) # Initialize perception_loss
    classifier_loss = torch.Tensor([-1]) # Initialize classifier_loss
    cross_entropy_loss = torch.Tensor([-1])  # for classification

    saved_original_test_image = False
    # =============================================================================

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        classifier1.train()

        Lambda = Lambda_s

        if Lambda == 0:
            # Give an early edge to training discriminator for Lambda = 0
            Lambda = compute_lambda_anneal(Lambda, epoch)

        for i, (x, y) in enumerate(dataloader):
            # Configure input
            image = x.to(device)

            # =============================================================================
            y = y.to(device)
            # =============================================================================

            noise = torch.randn(image.size()).mul_(args.noise_level/255.0).cuda()
            input = image + noise

            if i % n_cycles != 1:

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Unfreeze Distcriminator & Classifier
                free_params(discriminator)
                free_params(classifier1)

                # Freeze Generator
                frozen_params(generator)

                optimizer_D.zero_grad()

                output = generator(input)

                # Real images
                real_validity = discriminator(image)
                # Fake images
                fake_validity = discriminator(output)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, image.data, output.data)
                # Adversarial loss
                disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                disc_loss.backward()
                optimizer_D.step()

                # =============================================================================
                # ---------------------
                #  Train Classifier 
                # ---------------------

                optimizer_classifier1.zero_grad()

                # Pass reconstructed images through the classifier
                classifier_output = classifier1(output)
                classifier_loss = cross_entropy_criterion(classifier_output, y)

                classifier_loss.backward()
                optimizer_classifier1.step()
                # =============================================================================

            else: # if i % n_cycles == 1:

                # -----------------
                #  Train Generator
                # -----------------

                # Freeze Discriminator & Classifier
                frozen_params(discriminator)
                frozen_params(classifier1)
                
                # Unfreeze Generator
                free_params(generator)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                output = generator(input)

                real_validity = discriminator(image)
                fake_validity = discriminator(output)

                perception_loss = -torch.mean(fake_validity)  + torch.mean(real_validity)
                distortion_loss = criterion(input, output)

                # =============================================================================
                # Pass reconstructed images through the classifier
                classifier_output = classifier1(output)
                cross_entropy_loss = cross_entropy_criterion(classifier_output, y)

                loss = distortion_loss + Lambda*perception_loss + Lambda*cross_entropy_loss

                # loss = distortion_loss + Lambda*cross_entropy_loss
                # =============================================================================

                loss.backward()
                optimizer_G.step()

                # # Log scalar values
                # writer.add_scalar('Distortion Loss', distortion_loss.item(), epoch)
                # writer.add_scalar('Perception Loss', perception_loss.item(), epoch)

                # # Log images
                # writer.add_image('Goundtruth Image', vutils.make_grid(hr, normalize=True), epoch)
                # writer.add_image('Low Resolution Image', vutils.make_grid(lr, normalize=True), epoch)
                # writer.add_image('Super Resolution Image', vutils.make_grid(sr, normalize=True), epoch)


        # ---------------------
        # Evaluate losses on test set
        # ---------------------        
        if (epoch+1)%5 == 0:
            with torch.no_grad():
                generator.eval()
                discriminator.eval()
                classifier1.eval()

                mse_ori = 0
                mse_losses = 0
                perception_losses = 0

                cross_entropy_loss_avg = 0
                accuracy_avg = 0
                total_accuracy = 0

                for index, (data, label) in enumerate(dataloader_test):
                    image = data.to(device)
                    label = label.to(device)

                    noise = torch.randn(image.size()).mul_(args.noise_level/255.0).cuda()
                    input = image + noise
                    output = generator(input)

                    # Compute losses
                    mse_loss, perception_loss, cross_entropy_loss = evaluate_losses(input, output, label, discriminator, classifier1)

                    # Metrics
                    mse_losses += data.size(0) * mse_loss
                    perception_losses += data.size(0) * perception_loss
                    mse_ori += data.size(0) * torch.mean((output - image)**2)
                    cross_entropy_loss_avg += data.size(0) * cross_entropy_loss

                    # Accuracy metric
                    label_outputs = classifier1(output)

                    prediction = torch.max(label_outputs, 1)[1]
                    total_accuracy += (prediction == label).sum().item()

                    # =============================================================================  
                    if index == 0 and is_progress_interval(args, epoch):
                        save_image(unnormalizer(output.data[:120]), f"{experiment_path}/{epoch}_denosing.png", nrow=10, normalize=True)
                        save_image(unnormalizer(input.data[:120]), f"{experiment_path}/{epoch}_noise.png", nrow=10, normalize=True)
                        if not saved_original_test_image:
                            save_image(unnormalizer(image.data[:120]), f"{experiment_path}/{epoch}_original.png", nrow=10, normalize=True)
                            saved_original_test_image = True
                    # ============================================================================= 
                    
                ave_mse = mse_losses / test_set_size
                ave_per = perception_losses / test_set_size
                mse_a = mse_ori / test_set_size
                cross_entropy_loss_avg = cross_entropy_loss_avg / test_set_size
                accuracy_avg = total_accuracy / test_set_size

                print('distortion at epoch {} for mse {} and perception {} mse_ori {} cross entropy loss {} accuracy {}'.format(epoch, ave_mse, ave_per, mse_a, cross_entropy_loss_avg, accuracy_avg))
        
                # =============================================================================    
                with open(f"{experiment_path}/_perception_losses.csv", "a") as f:
                    f.write(f"{epoch},{ave_mse},{ave_per},{rate},{Lambda}\n")

                with open(f"{experiment_path}/_cross_entropy_losses.csv", "a") as f:
                    f.write(f"{epoch},{ave_mse},{cross_entropy_loss_avg},{rate},{Lambda}\n")

                with open(f"{experiment_path}/_accuracy.csv", "a") as f:
                    f.write(f"{epoch},{ave_mse},{accuracy_avg},{rate},{Lambda}\n")
                # =============================================================================
        
        lr_scheduler_G.step()
        lr_scheduler_D.step()
    
    # Training loop ends here and save the model
    torch.save(generator.state_dict(), os.path.join(weight_path, f'model_{latent_dim}.pth'))


if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="path to datasets. (default:./data)")
    parser.add_argument("--dataset", type=str, default="svhn",
                        help="dataset name. (default:`svhn`)"
                            "Option: [mnist, svhn, usps ]")
    parser.add_argument("--log_dir", type=str, default="./data")
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("-b", "--batch-size", default=64, type=int,
                        metavar="N",
                        help="mini-batch size (default: 1), this is the total "
                            "batch size of all GPUs on the current node when "
                            "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--cuda", default = True)
    parser.add_argument("--image_size", type=int, default=32,
                        help="size of the data crop (squared assumed). (default:32)")
    parser.add_argument("--model_save_dir", default="./outputs",
                        help="folder to output images. (default:`./outputs`).")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    
    parser.add_argument("--experiment_path", type=str, help="name of the subdirectory to save")
    parser.add_argument("--mode", type=str, default="base", help="super_res or dn mode")
    parser.add_argument("--progress_intervals", type=int, default=-1, help="periodically show progress of training")

    ## training setting
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--ql", default= True)
    parser.add_argument("--common", default= True)
    parser.add_argument("--stochastic", default=True)
    parser.add_argument("--adv_weight", type = int, default=0.001)
    parser.add_argument("--n_critic", type = int, default=1)
    parser.add_argument("--noise_level", type = int, default=20)

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device]: {device}")

    if args.mode == "dn":
        train_dn(args, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")