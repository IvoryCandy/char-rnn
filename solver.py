import math

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CharRNN
from data import TextDataset, TextConverter


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.convert = None
        self.model = None
        self.optimizer = None
        self.criterion = self.get_loss
        self.meter = AverageValueMeter()
        self.train_loader = None

        self.get_data()
        self.get_model()
        self.get_optimizer()

    def get_data(self):
        self.convert = TextConverter(self.args.txt, max_vocab=self.args.max_vocab)
        dataset = TextDataset(self.args.txt, self.args.len, self.convert.text_to_arr)
        self.train_loader = DataLoader(dataset, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def get_model(self):
        self.model = CharRNN(self.convert.vocab_size, self.args.embed_dim, self.args.hidden_size, self.args.num_layers, self.args.dropout, self.args.cuda)
        if self.args.cuda:
            self.model.cuda()
            cudnn.benchmark = True

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.optimizer = ScheduledOptim(optimizer)

    @staticmethod
    def get_loss(score, label):
        return nn.CrossEntropyLoss()(score, label.view(-1))

    def save_checkpoint(self, epoch):
        if (epoch + 1) % self.args.save_interval == 0:
            model_out_path = self.args.save_file + "epoch_{}_model.pth".format(epoch + 1)
            torch.save(self.model, model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))

    def save(self):
        model_out_path = self.args.save_file + "final_model.pth"
        torch.save(self.model, model_out_path)
        print("Final model saved to {}".format(model_out_path))

    @staticmethod
    def pick_top_n(predictions, top_n=5):
        top_predict_prob, top_predict_label = torch.topk(predictions, top_n, 1)
        top_predict_prob /= torch.sum(top_predict_prob)
        top_predict_prob = top_predict_prob.squeeze(0).cpu().numpy()
        top_predict_label = top_predict_label.squeeze(0).cpu().numpy()
        c = np.random.choice(top_predict_label, size=1, p=top_predict_prob)
        return c

    def train(self):
        self.meter.reset()
        self.model.train()
        for x, y in tqdm(self.train_loader):
            y = y.long()
            if self.args.cuda:
                x = x.cuda()
                y = y.cuda()
            x, y = Variable(x), Variable(y)

            # Forward.
            score, _ = self.model(x)
            loss = self.criterion(score, y)

            # Backward.
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradient.
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            self.optimizer.step()

            self.meter.add(loss.data[0])

        print('perplexity: {}'.format(np.exp(self.meter.value()[0])))

    def test(self):
        self.model.eval()
        begin = np.array([i for i in self.args.begin])
        begin = np.random.choice(begin, size=1)
        text_len = self.args.predict_len
        samples = [self.convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]

        if self.args.cuda:
            input_txt = input_txt.cuda()

        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]

        for i in range(text_len):
            out, init_state = self.model(model_input, init_state)
            prediction = self.pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(prediction))[None]
            if self.args.cuda:
                model_input = model_input.cuda()
            result.append(prediction[0])

        print(self.convert.arr_to_text(result))

    def predict(self):
        self.model.eval()
        samples = [self.convert.word_to_int(c) for c in self.args.begin]
        input_txt = torch.LongTensor(samples)[None]

        if self.args.cuda:
            input_txt = input_txt.cuda()

        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]

        for i in range(self.args.predict_len):
            out, init_state = self.model(model_input, init_state)
            prediction = self.pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(prediction))[None]
            if self.args.cuda:
                model_input = model_input.cuda()
            result.append(prediction[0])

        print(self.convert.arr_to_text(result))

    def run(self):
        for e in range(self.args.max_epoch):
            print('===> EPOCH: {}/{}'.format(e + 1, self.args.max_epoch))
            self.train()
            self.test()
            self.save_checkpoint(e)
        self.save()


class AverageValueMeter(object):
    """
    the meter tracker mainly focuses on mean and std
    """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.n = None
        self.sum = None
        self.var = None
        self.val = None
        self.mean = None
        self.std = None

        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
        else:
            self.mean = self.sum / self.n
            self.std = math.sqrt(
                (self.var - self.n * self.mean * self.mean) / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.std = np.nan


class ScheduledOptim(object):
    """A wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def lr_multi(self, multi):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= multi
        self.lr = self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr