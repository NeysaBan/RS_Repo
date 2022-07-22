from turtle import forward
# from sonnet import Dropout
import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, args, word_dim):

        super(CNN, self).__init__()

        self.kernel_count = args.kernel_count
        self.review_count = args.review_count


        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels = word_dim,
                out_channels = args.kernel_count, 
                kernel_size = args.kernel_size, 
                padding = (args.kernel_size - 1) // 2,  
            ),

            nn.ReLU(),


            nn.MaxPool2d(kernel_size = (1, args.review_length)),
            nn.Dropout(p = args.dropout)
        )

        self.fn = nn.Sequential(
            nn.Linear(self.kernel_count * self.review_count, args.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p = args.dropout)
        )

        self.initialize()
    
    def initialize(self):

        for m in self.conv:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                
            
        for m in self.fn:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    
    def forward(self, vec):
        
        latent = self.conv(vec.permute(0, 2, 1))
        latent = self.fn(latent.reshape(-1, self.kernel_count * self.review_count))
        
        return latent

class FactorizationMachine(nn.Module):
    def __init__(self, p, k): 

        super(FactorizationMachine, self).__init__()
    
        self.v = nn.Parameter(torch.zeros(p, k)) 
        self.linear = nn.Linear(p, 1, bias=True) 
        nn.init.kaiming_uniform_(self.linear.weight)
    

    def forward(self, x): 
        linear_part = self.linear(x) 
        
        inter_part1 = torch.mm(x, self.v) 
        inter_part2 = torch.mm(x**2, self.v**2) 

        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)

        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions


        return output.view(-1, 1) 
        
        #return output
        
class DeepCoNN(nn.Module):
    def __init__(self, args, word_emb):

        super(DeepCoNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(args, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(args, word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(args.cnn_out_dim * 2, 10)


    def forward(self, user_review, item_review):  

        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.reshape(new_batch_size, -1) 
        item_review = item_review.reshape(new_batch_size, -1)
        
        
        u_vec = self.embedding(user_review) 
        i_vec = self.embedding(item_review)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        
        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        return prediction, user_latent, item_latent
