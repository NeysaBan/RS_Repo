import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs):

        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.tensor(log_seqs)

        log_seqs = log_seqs.long().to(self.dev)
        seqs = self.item_emb(log_seqs).to(self.dev)

        
        seqs *= self.item_emb.embedding_dim ** 0.5

        # np.tile(array, res) -> (128, 200, 50)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)


        timeline_mask = (log_seqs == 0).bool().to(self.dev)


        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            ## seqs->(200, 128, 50)
            seqs = torch.transpose(seqs, 0, 1)

            Q = self.attention_layernorms[i](seqs)

            # Q = self.attention_layernorms[i](seqs), K = seqs, V = seqs
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        

        ## (128, 200, 50)
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        #pos_embs = self.item_emb(torch.cuda.LongTensor(pos_seqs).to(self.dev))
        pos_embs = self.item_emb(pos_seqs.long().to(self.dev))

        #neg_embs = self.item_emb(torch.cuda.LongTensor(neg_seqs).to(self.dev))
        neg_embs = self.item_emb(neg_seqs.long().to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        ## (128, 200, 50) -> (128, -1, 50)即(128, 50)
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        ## (128, 101) -> (128, 101, 50) 
        item_embs = self.item_emb(torch.cuda.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        
        ## (128, 101, 50) * (128, 50) -> (128, 101)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
