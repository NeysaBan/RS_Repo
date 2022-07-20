import torch
from torch import nn
import sys



class NCF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim ,  dropout, model_name, num_layers, GMF_model = None, MLP_model = None ):
        super(NCF, self).__init__()

        self.GMF_model = GMF_model
        self.MLP_model = MLP_model


        self.embed_user_GMF = nn.Embedding(user_num , embedding_dim)
        self.embed_item_GMF = nn.Embedding(item_num, embedding_dim)
        self.embed_user_MLP = nn.Embedding(user_num, embedding_dim * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, embedding_dim * (2 ** (num_layers - 1)))

        
        MLP_modules = []
        for i in range(num_layers):
            ### ???
            input_size = embedding_dim * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p = dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        self.model = model_name
        if self.model in ['MLP', 'GMF']:
            predict_size = embedding_dim
        else:
            predict_size = embedding_dim * 2
        ########
        self.relu = nn.ReLU()
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()
        

    def _init_weight_(self):
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std =0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std =0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std =0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std =0.01)


            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                        a = 1, nonlinearity='relu')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
                    
        else:
            self.embed_user_GMF.weight.data.copy_(
                    self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)

            self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
            
            self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight
            ], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias  + \
                            self.MLP_model.predict_layer.bias
            

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def forward(self, user, item, rating):

        if not self.model == 'MLP' : ## model = GMF, NeuMF_end, NeuMF_pre
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            #output_GMF = embed_user_GMF * embed_item_GMF #元素积
            output_GMF = torch.mul(embed_user_GMF, embed_item_GMF) ## 

        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), 1)
            output_MLP = self.MLP_layers(interaction)


        if self.model == 'GMF':
            concat = output_GMF
        
        elif self.model == 'MLP':
            concat = output_MLP

        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        out = self.predict_layer(concat)

        if (self.model == 'GMF') | (self.model == 'NeuMF-pre'):
            out = self.relu(out)

        return {"prediction":out, "labels":rating}