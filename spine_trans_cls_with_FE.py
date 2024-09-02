import os
import sys
sys.path.append(os.path.abspath('./'))
from random import seed
import dataset
from helpers import *
from cls_models.spine_transformer import *
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
torch.cuda.empty_cache()
import json
import csv
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
ROOT_DIR = "./"

def get_activation(activations, name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def train(training_dataset, testing_dataset, args, device):
    training_dataset_loader = dataset.spine_dataset_loader(training_dataset, args)
    start_epoch = 0
    tqdm_epoch = range(start_epoch, args['EPOCHS']+1)

    feature_dim = 512
    num_classes = 2
    model = SaliencyModulatedTransformer(feature_dim, num_classes)
    model = model.to(device)
    model.train()
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    losses = []
    iters = len(training_dataset) // args['Batch_Size']


    activations = {}
    features_list = []
    labels_list = []
    # model.layer4[1].conv2.register_forward_hook(get_activation(activations, 'last_conv'))
    # model.vit.blocks[-1].norm1.register_forward_hook(get_activation(activations, 'features'))
    # model.vit.encoder.layers.encoder_layer_11.ln_2.register_forward_hook(get_activation(activations, 'features'))

    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []
        print('epoch:', epoch)
        for i, data in enumerate(training_dataset_loader):
            if i % 40 == 0:
                print(i, "/", iters)
            images = data["images"].to(device)
            sals = data["sals"].to(device)
            labels = data["labels"].to(device)
            # Forward pass
            outputs = model(images, sals)
            # outputs = model(features, sals_features)
            loss = criterion(outputs.reshape(-1, 2), labels.reshape(-1))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss.append(loss.data.cpu())

            # if epoch % 5 == 0 and i == 0:
            #     feature = activations['features'].cpu()
            #     features_list.append(feature)
            #     labels_list.append(labels.cpu())

        losses.append(np.mean(mean_loss))
        print("loss:", losses[-1])

        if epoch % 1 == 0:
            print(str(epoch) + " epoch test:")
            model.eval()
            single_evaluation(model, testing_dataset, args, epoch, losses[-1], device)
            model.train()

            if args['t-SNE'] == "True":
                all_features = features_list[0].numpy()

                all_labels = labels_list[0].view(-1).numpy()

                mask = all_labels >= 0
                # all_features = all_features[mask]
                all_labels = all_labels[mask]
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(all_features.reshape(all_features.shape[0], -1))
                mask_flattened = mask.flatten()
                indices = np.nonzero(mask_flattened)
                features_2d = features_2d[indices]
                plt.figure(figsize=(10, 5))
                plt.scatter(features_2d[all_labels == 0, 0], features_2d[all_labels == 0, 1], color='red',
                            label='Class 0')
                plt.scatter(features_2d[all_labels == 1, 0], features_2d[all_labels == 1, 1], color='blue',
                            label='Class 1')
                plt.legend()
                plt.savefig(f'./training_outputs/t-SNE_ARGS={args["arg_num"]}/epoch={epoch}_tsne_visualization.png')
                # plt.show()

        if epoch % 5 == 0:
            # scheduler.step()
            save(model=model, args=args, epoch=epoch)


def single_evaluation(model, testing_dataset, args, epoch, loss, device):
    predicted_prob = []
    label = []
    testing_dataset_loader = dataset.spine_dataset_loader(testing_dataset, args, shuffle=False)
    iters = len(testing_dataset) // args['Batch_Size']
    with torch.no_grad():
        for i, data in enumerate(testing_dataset_loader):
            if i % 10 == 0:
                print(i, "/", iters)
            x = data["images"].to(device)
            s = data["sals"].to(device)
            l = data["labels"].to(device)
            # Forward pass
            output = model(x, s)
            probabilities = torch.sigmoid(output)[:, :, 1]
            predicted_prob.extend(probabilities.cpu().numpy().flatten())
            label.extend(l.cpu().numpy().flatten())

        label = np.array(label)
        predicted_prob = np.array(predicted_prob)
        mask = label >= 0
        label = label[mask]
        predicted_prob = predicted_prob[mask]
        auc = roc_auc_score(label, predicted_prob)
        predicted_label = (predicted_prob > 0.5).astype(int)
        accuracy = accuracy_score(label, predicted_label)
        precision = precision_score(label, predicted_label, zero_division=1)
        recall = recall_score(label, predicted_label)
        f1 = f1_score(label, predicted_label)

        print("AUC:", auc)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)

        with open(f'./training_outputs/ARGS={args["arg_num"]}_num={args["ex_num"]}_test={args["test_num"]}_results.csv', 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'loss', 'auc', 'accuracy', 'precision', 'recall', 'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch,
                             'loss': loss,
                             'auc': auc,
                             'accuracy': accuracy,
                             'precision': precision,
                             'recall': recall,
                             'f1': f1})

def save(model, args, epoch=0):
    model_save_path = f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/classifier_epoch={epoch}.pt'
    torch.save(model.state_dict(), model_save_path)

def main():
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")
    with open(f'{ROOT_DIR}configs/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    with open(f'./training_outputs/ARGS={args["arg_num"]}_num={args["ex_num"]}_test={args["test_num"]}_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'auc', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print(file, args)

    for directory in [f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}',
                      f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/',
                      f'./training_outputs/t-SNE_ARGS={args["arg_num"]}/'
                      ]:
        os.makedirs(directory, exist_ok=True)

    training_dataset, testing_dataset = dataset.diff_seg_datasets(ROOT_DIR, args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args["GPU"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(training_dataset, testing_dataset, args, device)

if __name__ == '__main__':
    seed(1)
    main()
