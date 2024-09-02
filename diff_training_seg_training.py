import collections
import csv
import sys
import time
from random import seed

import matplotlib.pyplot as plt
import seaborn as sns
from torch import optim
import torch.nn.functional as F

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel
from helpers import *
from model_unet import DiscriminativeSubNetwork

torch.cuda.empty_cache()
ROOT_DIR = "./"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(training_dataset, args):
    if args["channels"] != "":
        in_channels = args["channels"]
    else:
        in_channels = 1

    model_rec = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
            "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=in_channels
    )
    model_rec.to(device)

    model_seg = DiscriminativeSubNetwork(in_channels=2, out_channels=1)
    model_seg.to(device)
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
        {"params": model_rec.parameters(), "lr": args['lr'], "weight_decay": args['weight_decay'],
         "betas": (0.9, 0.999)},
        {"params": model_seg.parameters(), "lr": args['lr'], "weight_decay": args['weight_decay'],
         "betas": (0.9, 0.999)}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args['EPOCHS'] * 0.8, args['EPOCHS'] * 0.9], gamma=0.2,
                                               last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
    )

    start_epoch = 0
    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)

    start_time = time.time()
    losses = []
    rec_losses = []
    ssim_losses = []
    seg_losses = []
    noise_losses = []
    vlb = collections.deque([], maxlen=10)

    training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
    d_set_size = len(training_dataset) // args['Batch_Size']

    for epoch in tqdm_epoch:
        mean_loss = []
        mean_rec_loss = []
        mean_ssim_loss = []
        mean_seg_loss = []
        mean_noise_loss = []
        print('epoch:', epoch)
        for i, data in enumerate(training_dataset_loader):
            print('epoch:', epoch, "  ", i, "/", d_set_size)
            x = data["image"].to(device)
            input = data["input"].to(device)
            syn_mask = data["syn_mask"].to(device)
            box_unhealthy_mask = data["box_unhealthy_mask"].to(device)
            anomaly_mask = data["unhealthy_mask"].to(device)

            # anomaly_mask = torch.where(anomaly_mask > 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
            # anomaly_mask = anomaly_mask.float()

            # ---------------DDPM noise loss---------------------
            noise_loss, estimates = diffusion.p_loss(model_rec, input, args)
            gray_rec_seq = diffusion.forward_backward(
                model_rec, input,
                see_whole_sequence="half",
                t_distance=args["t_distance"], denoise_fn=args["noise_fn"]
            )
            gray_rec = gray_rec_seq[-1].to(device)
            x_t = gray_rec_seq[1].to(device)
            mse = (gray_rec - x).square()

            # -------seg_loss----------------------------
            joined_in = torch.cat((gray_rec * (1 - box_unhealthy_mask), x * (1 - box_unhealthy_mask)), dim=1)
            out_mask = torch.sigmoid(model_seg(joined_in))
            show_out_mask = (out_mask > 0.5).float()

            alpha = 0.25
            gamma = 2.0

            # seg_loss = loss_l2(out_mask, syn_mask)
            smooth_l1_loss = F.smooth_l1_loss(out_mask, syn_mask)
            focal_loss = torch.sum(-syn_mask * (1 - out_mask).pow(gamma) * torch.log(out_mask)
                                   - (1 - syn_mask) * out_mask.pow(gamma) * torch.log(1 - out_mask))
            lambda_f = 3.0
            seg_loss = smooth_l1_loss + lambda_f * focal_loss
            # -------rec_loss----------------------------
            healthy_gray_rec = gray_rec * (1 - box_unhealthy_mask)
            healthy_x = x * (1 - box_unhealthy_mask)
            rec_loss = loss_l2(healthy_gray_rec, healthy_x)

            # -------total loss-------------------------
            loss = noise_loss + rec_loss + seg_loss
            # ------------------------------------

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_rec.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model_seg.parameters(), 1)

            optimizer.step()

            mean_loss.append(loss.data.cpu())
            mean_rec_loss.append(rec_loss.data.cpu())
            mean_seg_loss.append(seg_loss.data.cpu())
            mean_noise_loss.append(noise_loss.data.cpu())

            if epoch % 5 == 0 and i <= 5:
                if args["draw_img"]:
                    out = {
                        'image': x[args['Batch_Size'] - 1:],
                        'input': input[args['Batch_Size'] - 1:],
                        'x_t': x_t[args['Batch_Size'] - 1:],
                        'gray_rec': gray_rec[args['Batch_Size'] - 1:],
                        'mse_heatmap': mse[args['Batch_Size'] - 1:],
                        'out_mask': out_mask[args['Batch_Size'] - 1:],
                        'unhealthy_masks': box_unhealthy_mask[args['Batch_Size'] - 1:],
                        "syn_mask": syn_mask[args['Batch_Size'] - 1:],
                    }

                    row_size = len(out.keys())
                    width = 2
                    fig, axes = plt.subplots(nrows=1, ncols=row_size, figsize=(width * row_size, width))
                    fig.subplots_adjust(wspace=0)
                    j = 0
                    for name, tensor in out.items():
                        array = tensor[0][0].cpu().detach().numpy()
                        if name == 'mse_heatmap' or name == 'out_mask':
                            scaled_mse = array * 255
                            sns.heatmap(data=scaled_mse, ax=axes[j], cmap='hot', cbar=False)
                        else:
                            axes[j].imshow(array, cmap='gray')
                        axes[j].set_title(name)
                        axes[j].axis('off')
                        j = j + 1
                    # plt.suptitle(str(data["id"][0]))
                    plt.rcParams['figure.dpi'] = 150
                    save_dir = f'./training_outputs/ARGS={args["arg_num"]}_t={args["t_distance"]}_num={args["ex_num"]}/epoch={epoch}'
                    try:
                        os.makedirs(save_dir)
                    except OSError:
                        pass
                    save_name = save_dir + "/iter_" + str(i) + "_" + str(data["id"][0]) + '.png'
                    plt.savefig(save_name)
                    print(save_name)
                    # print("training output has been saved!")
                    # plt.show()
                    plt.clf()

            # if i == 5:
            #     break

        losses.append(np.mean(mean_loss))
        rec_losses.append(np.mean(mean_rec_loss))
        ssim_losses.append(np.mean(mean_ssim_loss))
        seg_losses.append(np.mean(mean_seg_loss))
        noise_losses.append(np.mean(mean_noise_loss))

        print("loss:", losses[-1])
        print("rec_losses:", rec_losses[-1])
        print("seg_losses:", seg_losses[-1])
        print("noise_losses:", noise_losses[-1])

        if epoch % 10 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            vlb_terms = diffusion.calc_total_vlb(x, model_rec, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            print(
                f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                f" {np.mean(vlb):.4f}, "
                f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"est time remaining: {hours}:{mins:02.0f}\r"
            )
            training_csv = f'./training_outputs/ARGS={args["arg_num"]}_t={args["t_distance"]}_num={args["ex_num"]}/training_results.csv'
            with open(training_csv, 'w', newline='') as csvfile:
                fieldnames = ['Epoch', 'Total VLB', 'Mean Total VLB', 'Prior VLB', 'VB', 'X_0 MSE', 'MSE',
                              'Elapsed Time',
                              'Estimated Time Remaining']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(training_csv, 'a', newline='') as csvfile:
                fieldnames = ['Epoch', 'Total VLB', 'Mean Total VLB', 'Prior VLB', 'VB', 'X_0 MSE', 'MSE',
                              'Elapsed Time', 'Estimated Time Remaining']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                elapsed_time = int(time_taken / 3600), ((time_taken / 3600) % 1) * 60
                estimated_time_remaining = hours, mins

                writer.writerow({
                    'Epoch': epoch,
                    'Total VLB': vlb[-1],
                    'Mean Total VLB': np.mean(vlb),
                    'Prior VLB': vlb_terms['prior_vlb'].mean(dim=-1).cpu().item(),
                    'VB': torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item(),
                    'X_0 MSE': torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item(),
                    'MSE': torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item(),
                    'Elapsed Time': f"{elapsed_time[0]}:{elapsed_time[1]:02.0f}",
                    'Estimated Time Remaining': f"{estimated_time_remaining[0]}:{estimated_time_remaining[1]:02.0f}"
                })

        if epoch % 10 == 0:
            if epoch != args['EPOCHS']:
                scheduler.step()
                save(unet=model_rec, args=args, optimiser=optimizer, final=False, epoch=epoch, loss=losses[-1].item())
                seg_save(model=model_seg, args=args, optimiser=optimizer, final=False, epoch=epoch,
                         loss=losses[-1].item())
            else:
                scheduler.step()
                save(unet=model_rec, args=args, optimiser=optimizer, final=True)
                seg_save(model=model_seg, args=args, optimiser=optimizer, final=True, epoch=epoch)

    # evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)


def save(final, unet, optimiser, args, loss=0, epoch=0):
    if final:
        torch.save(
            {
                'n_epoch': args["EPOCHS"],
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                # "ema": ema.state_dict(),
                "args": args
                # 'loss': LOSS,
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/params-final.pt'
        )
    else:
        torch.save(
            {
                'n_epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args,
                # "ema": ema.state_dict(),
                'loss': loss,
            },
            f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/diff/diff_epoch={epoch}.pt'
        )


def seg_save(final, model, optimiser, args, loss=0, epoch=0):
    if final:
        torch.save(
            {
                'n_epoch': args["EPOCHS"],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/seg_params-final.pt'
        )
    else:
        torch.save(
            {
                'n_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args,
                'loss': loss,
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/seg/seg_epoch={epoch}.pt'
        )


def diff_seg_detection(d_set, args):
    if args["channels"] != "":
        in_channels = args["channels"]
    else:
        in_channels = 1

    diff_output, seg_output = load_parameters(args, device)

    model_rec = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
    )
    model_rec.load_state_dict(diff_output["model_state_dict"])
    model_rec.to(device)
    model_rec.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=2, out_channels=1)
    model_seg.load_state_dict(seg_output["model_state_dict"])
    model_seg.to(device)
    model_seg.eval()

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
    )

    loader = dataset.init_dataset_loader(d_set, args)
    d_set_size = len(d_set) // args['Batch_Size']

    for i, data in enumerate(loader):
        print(i, '/', d_set_size)
        x = data["image"].to(device)
        mask = data["mask"].to(device)
        unhealthy_mask = data["unhealthy_mask"].to(device)
        mask[mask == 1.0] = 1
        mask[mask != 1.0] = 0
        save_dir = data["save_dir"][0]
        data_id = data["id"][0]
        print(save_dir)

        if not data_id == 'healthy_215':
            continue

        # npy_name = f'ARGS={args["arg_num"]}_t={args["t_distance"]}_num={args["ex_num"]}_saliency_map.npy'
        # mask_path = os.path.join(save_dir, npy_name)
        # if os.path.exists(mask_path):
        #     print("exist!")
        #     continue

        gray_rec_seq = diffusion.forward_backward(
            model_rec, x,
            see_whole_sequence="half",
            t_distance=args["t_distance"], denoise_fn=args["noise_fn"]
        )
        gray_rec = gray_rec_seq[-1].to(device)
        x_t = gray_rec_seq[1].to(device)
        mse = (gray_rec - x).square()

        joined_in = torch.cat((gray_rec, x), dim=1)
        out_mask = torch.sigmoid(model_seg(joined_in))

        save_out_mask = out_mask.cpu().detach().numpy()
        npy_path = save_dir + f'/ARGS={args["arg_num"]}_t={args["t_distance"]}_num={args["ex_num"]}_saliency_map.npy'
        np.save(npy_path, save_out_mask)

        save_heatmap_data = save_out_mask[0, 0]
        plt.figure(figsize=(8, 6))
        sns.heatmap(save_heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
        plt.axis('off')
        output_path = save_dir + f'/ARGS={args["arg_num"]}_t={args["t_distance"]}_num={args["ex_num"]}_saliency_map.png'
        # plt.show()
        plt.savefig(output_path)

        if args["draw_img"]:
            out = {'image': x,
                   'x_t': x_t,
                   'gray_rec': gray_rec,
                   'mse_heatmap': mse,
                   # 'result_mse_heatmap': result,
                   # 'out_mask': out_mask,
                   'out_mask_heatmap': out_mask,
                   'out_mask_sal': out_mask,
                   # "image_with_boxes": image_with_boxes,
                   # 'show_out_mask': show_out_mask,
                   'unhealthy_mask': unhealthy_mask
                   }

            row_size = len(out.keys())
            width = 2
            fig, axes = plt.subplots(nrows=1, ncols=row_size, figsize=(width * row_size, width))
            fig.subplots_adjust(wspace=0)
            j = 0
            for name, tensor in out.items():
                if name == 'mse_heatmap' or name == 'out_mask_heatmap' or name == 'show_out_mask' or name == 'result_mse_heatmap':
                    array = tensor[0][0].cpu().detach().numpy()
                    scaled_mse = array * 255
                    sns.heatmap(data=scaled_mse, ax=axes[j], cmap='hot', cbar=False)
                elif name == "out_mask_sal":
                    array = tensor[0][0].cpu().detach().numpy()
                    img = x[0][0].cpu().detach().numpy()
                    cam_image = show_cam_on_image(img, array, use_rgb=True)
                    # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                    axes[j].imshow(cam_image)
                elif name == "image_with_boxes":
                    array = tensor[0].cpu().detach().numpy()
                    array = np.transpose(array, (1, 2, 0))
                    axes[j].imshow(array)
                else:
                    array = tensor[0][0].cpu().detach().numpy()
                    axes[j].imshow(array, cmap='gray')
                axes[j].set_title(name)
                axes[j].axis('off')
                j = j + 1
            # plt.suptitle(str(data["id"][0]))
            plt.rcParams['figure.dpi'] = 150
            out_save_dir = f'./diffusion-detection-images/ARGS={args["arg_num"]}_t={args["t_distance"]}_num={args["ex_num"]}/'
            if 'Train' in save_dir:
                out_save_dir = out_save_dir + f'Train{args["ex_num"]}/'
            else:
                out_save_dir = out_save_dir + f'Test{args["ex_num"]}/'
            try:
                os.makedirs(out_save_dir)
            except OSError:
                pass
            out_save_name = out_save_dir + str(data_id) + '.png'
            plt.savefig(out_save_name)
            print(out_save_name)
            # plt.show()
            plt.clf()
            plt.close()


def main():
    if len(sys.argv[1:]) > 0:
        file = sys.argv[-1]
    else:
        raise ValueError("Missing file argument")

    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    elif 'ARG_NUM=' in file:
        file = f"args{file.split('=')[-1]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args
    with open(f'{ROOT_DIR}configs/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    # make arg specific directories
    for i in [f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/seg',
              f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/diff',
              ]:
        try:
            os.makedirs(i)
        except OSError:
            pass

    print(file, args)

    # load my vertebrae dataset
    training_dataset, testing_dataset = dataset.diff_seg_datasets(ROOT_DIR, args)

    if args["mode"] == 'train':
        train(training_dataset, args)
    else:
        diff_seg_detection(training_dataset, args)
        # diff_seg_detection(testing_dataset, args)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    main()
