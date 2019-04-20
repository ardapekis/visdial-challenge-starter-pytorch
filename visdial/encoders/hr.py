import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utilities as utils


class HierarchicalRecurrentEncoder(nn.Module):
    def __init__(self,
                 config,
                 *args,
                 **kwargs):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.embed_size = config["embed_size"]
        self.rnn_hidden_size = config["rnn_hidden_size"]
        self.num_layers = config["num_layers"]
        use_im = config["use_im"]
        assert self.num_layers > 1, "Less than 2 layers not supported!"
        if use_im:
            self.use_im = use_im if use_im != True else 'early'
        else:
            self.use_im = False
        self.img_embed_size = config["img_embed_size"]
        self.img_feature_size = config["img_feature_size"]
        self.num_rounds = config["num_rounds"]
        self.dropout = config["dropout"]
        self.is_answerer = config["is_answerer"]
        if "start_token" in config:
            self.start_token = config["start_token"]
        else:
            self.start_token = None

        if "end_token" in config:
            self.end_token = config["end_token"]
        else:
            self.end_token = None

        # modules
        self.word_embed = nn.Embedding(
            self.vocab_size, self.embed_size, padding_idx=0)

        # question encoder
        # image fuses early with words
        if self.use_im == 'early':
            ques_input_size = self.embed_size + self.img_embed_size
            dialog_input_size = 2 * self.rnn_hidden_size
            self.img_net = nn.Linear(self.img_feature_size, self.img_embed_size)
            self.img_embed_dropout = nn.Dropout(self.dropout)
        elif self.use_im == 'late':
            ques_input_size = self.embed_size
            dialog_input_size = 2 * self.rnn_hidden_size + self.img_embed_size
            self.img_net = nn.Linear(self.img_feature_size, self.img_embed_size)
            self.img_embed_dropout = nn.Dropout(self.dropout)
        elif self.is_answerer:
            ques_input_size = self.embed_size
            dialog_input_size = 2 * self.rnn_hidden_size
        else:
            dialog_input_size = self.rnn_hidden_size

        if self.is_answerer:
            self.ques_rnn = nn.LSTM(
                ques_input_size,
                self.rnn_hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=0)

        # history encoder
        self.fact_rnn = nn.LSTM(
            self.embed_size,
            self.rnn_hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0)

        # dialog rnn
        self.dialog_rnn = nn.LSTMCell(dialog_input_size, self.rnn_hidden_size)

    def _init_hidden(self, batch_size):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        some_tensor = self.dialog_rnn.weight_hh.data
        h = some_tensor.new(batch_size, self.dialog_rnn.hidden_size).zero_()
        c = some_tensor.new(batch_size, self.dialog_rnn.hidden_size).zero_()
        return (h, c)

    def observe(self,
                ques=None,
                ans=None,
                ques_lens=None,
                ans_lens=None):
        '''
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''
        if ques is not None:
            assert ques_lens is not None, "Questions lengths required!"
            ques, ques_lens = self.process_sequence(ques, ques_lens)
            self.question_tokens.append(ques)
            self.question_lens.append(ques_lens)
        if ans is not None:
            assert ans_lens is not None, "Answer lengths required!"
            ans, ans_lens = self.process_sequence(ans, ans_lens)
            self.answer_tokens.append(ans)
            self.answer_lengths.append(ans_lens)

    def process_sequence(self, seq, seq_len):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seq_len - 1

    def embed_fact(self, batch):
        '''Embed facts i.e. question-answer pair'''
        # QA pairs
        ques_tokens, ques_lens = \
            self.question_tokens[fact_idx - 1], self.question_lens[fact_idx - 1]
        ansTokens, ans_lens = \
            self.answer_tokens[fact_idx - 1], self.answer_lengths[fact_idx - 1]

        qa_tokens = utils.concatPaddedSequences(
            ques_tokens, ques_lens, ansTokens, ans_lens, padding='right')
        qa = self.word_embed(qa_tokens)
        qa_lens = ques_lens + ans_lens
        qa_embed, states = utils.dynamicRNN(
            self.fact_rnn, qa, qa_lens, returnStates=True)
        fact_embed = qa_embed
        fact_rnnstates = states
        self.fact_embeds.append((fact_embed, fact_rnnstates))

    def embed_question(self, qIdx):
        '''Embed questions'''
        ques_in = self.question_embeds[qIdx]
        ques_lens = self.question_lens[qIdx]
        if self.use_im == 'early':
            image = self.image_embed.unsqueeze(1).repeat(1, ques_in.size(1), 1)
            ques_in = torch.cat([ques_in, image], 2)
        q_embed, states = utils.dynamicRNN(
            self.ques_rnn, ques_in, ques_lens, returnStates=True)
        ques_rnnstates = states
        self.question_rnn_states.append((q_embed, ques_rnnstates))

    def concat_dialog_rnn_input(self, hist_idx):
        curr_ins = [self.fact_embeds[hist_idx][0]]
        if self.is_answerer:
            curr_ins.append(self.question_rnn_states[hist_idx][0])
        if self.use_im == 'late':
            curr_ins.append(self.image_embed)
        hist_t = torch.cat(curr_ins, -1)
        self.dialog_rnn_inputs.append(hist_t)

    def embed_dialog(self, dialog_idx):
        if dialog_idx == 0:
            hPrev = self._init_hidden()
        else:
            hPrev = self.dialog_hiddens[-1]
        inpt = self.dialog_rnn_inputs[dialog_idx]
        hNew = self.dialog_rnn(inpt, hPrev)
        self.dialog_hiddens.append(hNew)

    def forward(self, batch):
        '''
        Accepts batch, dict of tensors:
        - img_ids:      (batch_size,)
        - img_feat:     (batch_size, regions, img_feature_size)
        - ques:         (batch_size, 10, max_seq_len)
        - hist:         (batch_size, 10, max_seq_len * 2 * 10)
        - ans_in:       (batch_size, 10, max_seq_len)
        - ans_out:      (batch_size, 10, max_seq_len)
        - ques_len:     (batch_size, 10)
        - hist_len:     (batch_size, 10)
        - ans_len:      (batch_size, 10)
        - num_rounds:   (batch_size,)
        - opt:          (batch_size, 10, 100)
        - opt_len:      (batch_size, 10, 100)
        - ans_ind:      (batch_size, 10)
        Returns:
            A tuple of tensors (H, C) each of shape (batch_size, rnn_hidden_size)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''
        batch_size = batch["img_ids"].size(0)
        question_embeds = self.word_embed(batch["ques"])
        answer_embeds = self.word_embed(batch["ans_in"])

        # Infer any missing facts
        while len(self.fact_embeds) <= round:
            fact_idx = len(self.fact_embeds)
            self.embed_fact(fact_idx)

        # Embed any un-embedded questions (A-Bot only)
        if self.is_answerer:
            while len(self.question_rnn_states) <= round:
                qIdx = len(self.question_rnn_states)
                self.embed_question(qIdx)

        # Concat facts and/or questions (i.e. history) for input to dialog_rnn
        while len(self.dialog_rnn_inputs) <= round:
            hist_idx = len(self.dialog_rnn_inputs)
            self.concat_dialog_rnn_input(hist_idx)

        # Forward dialog_rnn one step
        while len(self.dialog_hiddens) <= round:
            dialog_idx = len(self.dialog_hiddens)
            self.embed_dialog(dialog_idx)

        # Latest dialog_rnn hidden state
        dialog_hidden = self.dialog_hiddens[-1][0]

        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for num_layers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (ques_rnn)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (ques_rnn)
              Layer 1 : DialogRNN hidden state (dialog_rnn)

        If Q-Bot:
            C_link == Fact encoding RNN cell state (fact_rnn)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (fact_rnn)
                Layer 1 : DialogRNN hidden state (dialog_rnn)
        '''
        if self.is_answerer:
            ques_rnnstates = self.question_rnn_states[-1][1]  # Latest ques_rnn states
            C_link = ques_rnnstates[1]
            H_link = ques_rnnstates[0][:-1]
            H_link = torch.cat([H_link, dialog_hidden.unsqueeze(0)], 0)
        else:
            fact_rnnstates = self.fact_embeds[-1][1]  # Latest fact_rnn states
            C_link = fact_rnnstates[1]
            H_link = fact_rnnstates[0][:-1]
            H_link = torch.cat([H_link, dialog_hidden.unsqueeze(0)], 0)

        return H_link, C_link
