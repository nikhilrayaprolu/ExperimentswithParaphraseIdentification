
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import attention.transformer.Constants as Constants
from attention.transformer.Modules import BottleLinear as Linear
from attention.transformer.Layers import EncoderLayer, DecoderLayer

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,data, args, n_layers=6, n_head=6, d_k=50, d_v=50,
            d_word_vec=300, d_model=300, d_inner_hid=600, dropout=0.1, ):

        super(Encoder, self).__init__()

        self.d_model = d_model

        self.position_enc = nn.Embedding(100, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(100, d_word_vec)
        self.src_word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        #print(data.TEXT.vocab.vectors)
        self.src_word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        self.src_word_emb.weight.requires_grad = False
        self.position_enc.weight.requires_grad = False
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        # Word embedding look up
        #print(src_seq)
        enc_input = self.src_word_emb(src_seq)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,
        
   


# In[2]:


import argparse

import torch
from torch import nn
from torch.autograd import Variable

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
#from attention.transformer.Models import Encoder

def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda()
                char_h = char_h.cuda()

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = model(**kwargs)
        pred = pred.view(-1,2)
        batch_loss = criterion(pred, batch.label)
        #print(batch_loss.shape)
        loss += batch_loss.data[0]
        #print()
        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc


def load_model(args, data):
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.cuda()

    return model




# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
    
class Siamese(nn.Module):
    def __init__(self, args, data, use_attention = False):
        super(Siamese, self).__init__()

        self.args = args
        self.d = self.args.word_dim + int(self.args.use_char_emb) * self.args.char_hidden_size
        self.l = self.args.num_perspective

        # ----- Word Representation Layer -----
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        
        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # no fine-tuning for word vectors
        self.word_emb.weight.requires_grad = False
        self.trainingtype = args.training
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = Attention(self.args.hidden_size*2)
        
        self.char_LSTM = nn.LSTM(
            input_size=self.args.char_dim,
            hidden_size=self.args.char_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True)

        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=self.d,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.args.hidden_size*2,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.encoder = Encoder(
            data, args)
        
        self.Ws = nn.Parameter(torch.rand(self.args.hidden_size*2,self.args.hidden_size*2))
        self.Us = nn.Parameter(torch.rand(self.args.hidden_size*2,self.args.hidden_size*2))
        self.bs = nn.Parameter(torch.rand(self.args.hidden_size*2))
        
        # ----- Prediction Layer -----
        self.pred_fc1 = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size * 1)
        self.pred_fc2 = nn.Linear(self.args.hidden_size * 1, self.args.class_size)

        self.reset_parameters()

    def reset_parameters(self):
        # ----- Word Representation Layer -----
        nn.init.uniform(self.char_emb.weight, -0.005, 0.005)
        # zero vectors for padding
        self.char_emb.weight.data[0].fill_(0)

        # <unk> vectors is randomly initialized
        nn.init.uniform(self.word_emb.weight.data[0], -0.1, 0.1)

        nn.init.kaiming_normal(self.char_LSTM.weight_ih_l0)
        nn.init.constant(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.char_LSTM.weight_hh_l0)
        nn.init.constant(self.char_LSTM.bias_hh_l0, val=0)

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0)
        nn.init.constant(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0)
        nn.init.constant(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc1.bias, val=0)

        nn.init.uniform(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout, training=self.training)

    def forward(self, **kwargs):
        p = self.word_emb(kwargs['p'])
        h = self.word_emb(kwargs['h'])
        #print(kwargs['p'].shape, kwargs['h'].shape)
        p_pos = (torch.arange(0,p.shape[1]).view(1,-1).expand(p.shape[0],p.shape[1])).type(torch.LongTensor).cuda()
        h_pos = (torch.arange(0,h.shape[1]).view(1,-1).expand(h.shape[0],h.shape[1])).type(torch.LongTensor).cuda()
        p_enc_output, *_ = self.encoder(kwargs['p'], p_pos)
        h_enc_output, *_ = self.encoder(kwargs['h'], h_pos)
        #print(p_enc_output.shape, h_enc_output.shape)
        p_enc_output_mean = torch.mean(p_enc_output, 1, True)
        h_enc_output_mean = torch.mean(h_enc_output, 1, True)
        #print(p_enc_output_mean.shape, h_enc_output_mean.shape)
        x = torch.cat(
            [p_enc_output_mean,h_enc_output_mean], dim=-1)
        
        #print(x.shape)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = F.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)
        return x

        
        
        #x = self.dropout(x)

        # ----- Prediction Layer -----
        #x = F.tanh(self.pred_fc1(x))
        #x = self.dropout(x)
        #x = self.pred_fc2(x)
        #print(x.shape)
        return 1


# In[4]:


import argparse
import copy
import os
import torch

from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora



def train(args, data):
    model = (Siamese(args, data,use_attention = True))
    if args.gpu > -1:
        model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', str(present_epoch + 1))
        last_epoch = present_epoch

        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        #print(s1.shape, s2.shape)

        # limit the lengths of input sentences up to max_sent_len
        if args.max_sent_len >= 0:
            if s1.size()[1] > args.max_sent_len:
                s1 = s1[:, :args.max_sent_len]
            if s2.size()[1] > args.max_sent_len:
                s2 = s2[:, :args.max_sent_len]

        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda()
                char_h = char_h.cuda()

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = (model(**kwargs))
        
        optimizer.zero_grad()
        #print(pred.shape, batch.label.shape)
        batch_loss = criterion(pred.view(-1,2), batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()
        del pred
        del batch_loss
        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_acc = test(model, args, data, mode='dev')
            test_loss, test_acc = test(model, args, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('acc/dev', dev_acc, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/test', test_acc, c)

            print('train loss: '+ str(loss) +' / dev loss: '+ str(dev_loss) + '/ test loss:' + str(test_loss) +
                  ' / dev acc:' + str(dev_acc) + 'test acc:' + str(test_acc))

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    writer.close()
    print('max dev acc:'+ str(max_dev_acc) + '/ max test acc: ' + str(max_test_acc))

    return best_model


def main():
    import sys
    sys.argv = ['foo']
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--data-type', default='Quora', help='available: SNLI or Quora')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=15, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=300, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--max-sent-len', default=100, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--training', default=0, type=int)
    args = parser.parse_args()
    print(args.training)
    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), 'saved_models/BIBPM_'+args.data_type+'_'+args.model_time+'train'+args.training+'.pt')
    print('training finished!')


if __name__ == '__main__':
    main()

