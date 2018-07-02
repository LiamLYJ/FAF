import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from nets import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tensorboardX import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):
    writer = SummaryWriter('./log')
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader
    data_loader = get_loader(args.data_file, transform, args.batch_size, shuffle =
                             True, num_workers = args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, args.num_class, args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # dummy_encoder = torch.rand(1, 3, args.crop_size, args.crop_size)
    # writer.add_graph(encoder, dummy_encoder)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        # images shape [batch_size, sequence, w,h,c]
        # lables shape [batch_size, sequence]
        for i, (images, labels) in enumerate(data_loader):

            # print ('images: ', images.shape)
            # print ('labels: ', labels.shape)

            # Set mini-batch dataset
            images = images.to(device)
            labels = labels.to(device)
            lengths = [labels.shape[1]] * args.batch_size
            targets = pack_padded_sequence(labels, lengths, batch_first=True)[0].long()

            # Forward, backward and optimize
            # features shape
            features = []
            assert (labels.shape[1] == images.shape[1])
            # print ('labels.shape: ', labels.shape)
            # print ('images.shape: ', images.shape)
            sequence_length = labels.shape[1]
            # print ('sequence_length: ', sequence_length)
            for index in range(sequence_length):
                features.append(encoder(images[:,index,:,:,:]))
            features = torch.stack(features, 1)
            outputs = decoder(features, lengths)
            # print ('feaute shape: ', features.shape)
            # print ('lengths: ', lengths)
            # print ('shape of oputputs: ', outputs.shape)
            # print ('shape of targets: ', targets.shape)
            # print ('outputs: ', outputs)
            # print ('targets: ', targets)
            # print ('*********************')
            loss = criterion(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                writer.add_scalar('loss',loss.item(), i + epoch * total_step)

            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                print ('model saved once : epoch: %d, iteration : %d'%(epoch + 1, i+1))

    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--image_dir', type=str, default='data/images', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    parser.add_argument('--data_file', type=str, default='./data/tmp.pkl', help='pkl file ')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--num_class', type=int , default=7, help='number of lens positions ')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
