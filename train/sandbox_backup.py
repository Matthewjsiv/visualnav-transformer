import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
import torch.nn.functional as F

from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.models.unet_2d_condition import UNet2DConditionModel

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import Vanilla_NoMaD, NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn, Vanilla_NoMaD_ViNT
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)

def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_trajectories: int,
    num_samples: int,
    device: torch.device,
):

    # print(batch_obs_images.requires_grad)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images)
    # obs_cond = obs_cond.flatten(start_dim=1)
    # print(obs_cond.requires_grad)
    obs_cond = obs_cond.unsqueeze(1)

    # print(obs_cond.shape)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)
    # print(obs_cond.shape)
    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), action_dim, num_trajectories, pred_horizon), device=device)
    diffusion_output = noisy_diffusion_output

    # print(diffusion_output.shape)
    # print(obs_cond.requires_grad, diffusion_output.requires_grad)
    for k in noise_scheduler.timesteps[:]:
        # predict noise
        # print('here')
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        ).sample

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    return diffusion_output


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}


    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vanilla_nomad_vint":
            vision_encoder = Vanilla_NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib":
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"], kernel_size=3
            )

        twod_noise_pred_net = UNet2DConditionModel(
                sample_size=(10,50),
                in_channels=2,
                out_channels=2,
                encoder_hid_dim=config["encoding_size"]
            ).cuda()

        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = Vanilla_NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    elif config["model_type"] == "vanilla_nomad":
        vision_encoder = Vanilla_NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        # noise_pred_net = UNet2DConditionModel(
        #         sample_size=(config["num_trajectories"],config["num_waypoints"]),
        #         in_channels=2,
        #         out_channels=2,
        #         encoder_hid_dim=config["encoding_size"]
        #     )

        noise_pred_net = UNet2DConditionModel(
                sample_size=(config["num_trajectories"],config["num_waypoints"]),
                in_channels=2,
                out_channels=2,
                block_out_channels=[32,64,128,256],
                encoder_hid_dim=config["encoding_size"]
            )

        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = Vanilla_NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # print(vision_encoder)
        # print(')))))))))))))))))')
        # print(noise_pred_net)

    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    if config["model_type"] == "vint" or config["model_type"] == "gnm":
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:
        # train_eval_loop_nomad(
        #     train_model=config["train"],
        #     model=model,
        #     optimizer=optimizer,
        #     lr_scheduler=scheduler,
        #     noise_scheduler=noise_scheduler,
        #     train_loader=train_loader,
        #     test_dataloaders=test_dataloaders,
        #     transform=transform,
        #     goal_mask_prob=config["goal_mask_prob"],
        #     epochs=config["epochs"],
        #     device=device,
        #     project_folder=config["project_folder"],
        #     print_log_freq=config["print_log_freq"],
        #     wandb_log_freq=config["wandb_log_freq"],
        #     image_log_freq=config["image_log_freq"],
        #     num_images_log=config["num_images_log"],
        #     current_epoch=current_epoch,
        #     alpha=float(config["alpha"]),
        #     use_wandb=config["use_wandb"],
        #     eval_fraction=config["eval_fraction"],
        #     eval_freq=config["eval_freq"],
        # )

        B = 10
        costmap = torch.rand((B,1,120,120)).cuda()
        # TRAJ_LIB = np.load('../../traj_lib_testing/traj_lib.npy')[:,:,:2][::500][:10]
        # #
        # TRAJ_LIB = np.stack([TRAJ_LIB]*B)
        # # for i in range(10):
        # #     plt.plot(TRAJ_LIB[0,i,:,1],TRAJ_LIB[0,i,:,0])
        # # plt.show()
        # # print(TRAJ_LIB.shape)
        # # traj = torch.randn((B,2,10,78), device=device)
        # traj = torch.tensor(TRAJ_LIB,dtype=torch.float32).cuda().permute((0,3,1,2))
        # print(traj.dtype)


        # B = traj.shape[0]

        losses = []
        TRAJ_LIB = np.load('../../traj_lib_testing/traj_lib.npy')[:,:,:2][::500]
        traj_ids = np.random.permutation(len(TRAJ_LIB))[:10]
        for i in range(10):
            plt.plot(TRAJ_LIB[i,:,0],TRAJ_LIB[i,:,1])
        plt.show()

        for i in range(1000):
            #
            lib = np.stack([TRAJ_LIB[traj_ids]]*B)
            # for i in range(10):
            #     plt.plot(TRAJ_LIB[0,i,:,1],TRAJ_LIB[0,i,:,0])
            # plt.show()
            # print(TRAJ_LIB.shape)
            # traj = torch.randn((B,2,10,78), device=device)
            traj = torch.tensor(lib,dtype=torch.float32).cuda().permute((0,3,1,2))/30
            # print(traj.max())

            # Generate random goal mask
            obsgoal_cond = model("vision_encoder", obs_img=costmap)

            # Sample noise to add to actions
            noise = torch.randn(traj.shape, device=device)
            # print(noise.shape)
            # print(noise.dtype)
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps - 7,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_traj = noise_scheduler.add_noise(
                traj, noise, timesteps)

            # print(traj.shape, noisy_traj.shape)

            vnoisy_traj = noisy_traj.clone().cpu().numpy()
            # for i in range(10):
            #     plt.plot(vnoisy_traj[0,1,i,:],vnoisy_traj[0,0,i,:])
            # plt.show()

            # Predict the noise residual
            obsgoal_cond = obsgoal_cond.unsqueeze(1)
            noise_pred = model("noise_pred_net", sample=noisy_traj, timestep=timesteps, global_cond=obsgoal_cond).sample
            # print(noise_pred.mean(),noise.mean())

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            # diffusion_loss = F.mse_loss(noise_pred, noise)
            # print(diffusion_loss, otherloss)
            loss = 1.0 * diffusion_loss
            print(loss)
            losses.append(loss.item())
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if i % 100 == 0:
                model.eval()
                with torch.no_grad():
                    diffusion_output = model_output(
                        model,
                        noise_scheduler,
                        costmap[0].unsqueeze(0).detach(),
                        78,
                        2,
                        10,
                        1,
                        device,
                    )
                    diffusion_output = diffusion_output.permute(0,2,3,1)
                    diffusion_output = torch.flatten(diffusion_output,end_dim=1).cpu().numpy()
                    for i in range(10):
                        plt.plot(diffusion_output[i,:,0],diffusion_output[i,:,1])
                    plt.show()
                    # print(diffusion_output.shape)
                model.train()

        # plt.plot(losses)
        # plt.show()

        # Total loss
        # loss = alpha * dist_loss + (1-alpha) * diffusion_loss


        #
        # noise = torch.randn(action.shape, device=device)
        # timesteps = torch.randint(
        #     0, noise_scheduler.config.num_train_timesteps,
        #     (B,), device=device
        # ).long()
        #
        # # Add noise to the clean images according to the noise magnitude at each diffusion iteration
        # noisy_action = noise_scheduler.add_noise(
        #     action, noise, timesteps)
        #
        # # print(obsgoal_cond.shape)
        #
        # noise_pred = model("noise_pred_net",sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond.unsqueeze(0))

        print('here')

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="gnmv2", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
