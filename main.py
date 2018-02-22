import argparse
import torch
from solver import Trainer

parser = argparse.ArgumentParser()
# dataset setup
parser.add_argument('--txt', required=False, type=str, default='./dataset/poetry.txt', help='training content directory')
parser.add_argument('--len', required=False, type=int, default=20, help='length for each training content line')
parser.add_argument('--max_vocab', required=False, type=int, default=8000, help='max vocab length')
parser.add_argument('--begin', required=False, type=str, default='天青色等烟雨', help='begin characters of text')
parser.add_argument('--predict_len', required=False, type=int, default=50, help='length of prediction text')

# saving directories
parser.add_argument('--result_file', required=False, type=str, default='result.txt', help='directory to save result')
parser.add_argument('--save_file', required=False, type=str, default='./checkpoints/', help='directory to save checkpoints')
parser.add_argument('--save_interval', required=False, type=int, default=30, help='the frequency to save checkpoints in cases of epochs')

# generated context
parser.add_argument('--load_model', required=False, type=str, default='./checkpoints/CharRNN_final_model.pth', help='directory of the model to be loaded')
parser.add_argument('--write_file', required=False, type=str, default='./write_context.txt', help='directory of output context')

# visualization setup
parser.add_argument('--vis_dir', required=False, type=str, default='./visualization/', help='directory of visualization stuff')
parser.add_argument('--plot_interval', required=False, type=int, default=100, help='the frequency to plot in tensorboard in cases of epochs')

# model parameters
parser.add_argument('--embed_dim', required=False, type=int, default=512, help='the embedding dimension')
parser.add_argument('--hidden_size', required=False, type=int, default=512, help='the hidden size')
parser.add_argument('--num_layers', required=False, type=int, default=2, help='number of layers of net')
parser.add_argument('--dropout', required=False, type=float, default=0.5, help='the dropout rate')

# model hyper-parameters
parser.add_argument('--cuda', required=False, type=bool, default=True, help='whether using cuda')
parser.add_argument('--batch_size', required=False, type=int, default=128, help='the batch size')
parser.add_argument('--num_workers', required=False, type=int, default=4, help='number of workers, basically 4 * number of GPU')
parser.add_argument('--max_epoch', required=False, type=int, default=200, help='the max epochs')
parser.add_argument('--lr', required=False, type=float, default=1e-3, help='the learning rate')
parser.add_argument('--weight_decay', required=False, type=float, default=1e-4, help='the weight decay')

args = parser.parse_args()


def main():
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
        print("Since GPU is not available, model will run on CPU")

    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
