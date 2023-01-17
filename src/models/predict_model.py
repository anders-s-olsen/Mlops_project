import argparse
import sys

import torch
import torch.nn.functional as F
import torchvision
from vit_pytorch import ViT


def predict():
    parser = argparse.ArgumentParser(description="Evaluating arguments")
    parser.add_argument("load_model_from", type=str)
    parser.add_argument("load_input_from", type=str)
    parser.add_argument("-d", "--device")
    # add any additional argument that you want
    # args = parser.parse_args(sys.argv[1:])
    args, unknown = parser.parse_known_args()
    print(args)

    if args.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    print("Device: ", device)

    model = ViT(
        image_size=28,
        patch_size=4,
        num_classes=10,
        channels=1,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
    )
    model.load_state_dict(torch.load(args.load_model_from))
    model.to(device)
    model.eval()

    input = torchvision.io.read_image(
        args.load_input_from, torchvision.io.ImageReadMode.GRAY
    ).float()
    input = input.unsqueeze(0)
    input = torch.narrow(input, 2, 50, 28)
    input = torch.narrow(input, 3, 50, 28)
    print(input.shape)

    # Normalize data
    mean = input.mean(dim=(1, 2, 3), keepdim=True)
    std = input.std(dim=(1, 2, 3), keepdim=True)

    input = (input - mean) / std

    with torch.no_grad():
        preds = F.softmax(model(input.to(device)), dim=1)
        pred = preds.argmax(dim=-1)
        print(
            f"The predicted label is: {pred.item()} with probability {preds.max().item():.2f}"
        )
    #     for data, target in data_load:
    #         output = F.log_softmax(model(data), dim=1)
    #         loss = F.nll_loss(output, target, reduction='sum')
    #         _, pred = torch.max(output, dim=1)

    #         tloss += loss.item()
    #         csamp += pred.eq(target).sum()
    #     wandb.log({"validation_loss":tloss.item()})

    # aloss = tloss / samples
    # loss_val.append(aloss)
    # print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
    #       '  Accuracy:' + '{:5}'.format(csamp) + '/' +
    #       '{:5}'.format(samples) + ' (' +
    #       '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')


#########################

### Load data
# tr_data = torch.load("data/processed/train.pt")
# tr_load = torch.utils.data.DataLoader(tr_data,batch_size=64,shuffle=True)
# ts_data = torch.load("data/processed/x_test.pt")
# ts_load = torch.utils.data.DataLoader(ts_data,batch_size=64,shuffle=False)

# N_EPOCHS = 1

# start_time = time.time()
# model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1,
#             dim=64, depth=6, heads=8, mlp_dim=128)
# optimz = optim.Adam(model.parameters(), lr=lr)

# trloss_val, tsloss_val = [], []
# for epoch in range(1, N_EPOCHS + 1):
#     print('Epoch:', epoch)
#     train_iter(model, optimz, tr_load, trloss_val)
#     evaluate(model, ts_load, tsloss_val)
# torch.save(model.state_dict(),'models/trained_model.pt')
# print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


if __name__ == "__main__":
    predict()
