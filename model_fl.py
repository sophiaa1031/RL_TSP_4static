import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class Actor(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.,iteration=10):
        super(Actor, self).__init__()
        self.update_fn = update_fn
        self.mask_fn = update_fn
        # Define the encoder & decoder models
        self.static_encoder = nn.Linear(static_size, hidden_size)
        self.dynamic_encoder = nn.Linear(dynamic_size, hidden_size)
        self.actor1_select = nn.Linear(hidden_size*2, 2)
        self.actor2_quant = nn.Linear(hidden_size*2, 32)
        self.actor3_b = nn.Linear(hidden_size*2, 1)
        self.iteration = iteration

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        max_steps = self.iteration# if self.mask_fn is None else 50

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        print ("static.shape",static.shape)


        whether_select_lst, quant_select_lst, b_lst = [], [], []
        whether_select_logp, quant_select_logp, b_logp = [], [], []
        for step in range(max_steps-1):

            static_hidden = self.static_encoder(static[:,:,:,step])#[]
            dynamic_hidden = self.dynamic_encoder(dynamic[:,:,:,step])
            state = torch.cat((static_hidden, dynamic_hidden), 2)#[batch_size,num_cars,2*hidden_size]
            whether_select = self.actor1_select(state) # (batch_size,num_cars,2)
            quant_select = self.actor2_quant(state) #(batch_size,num_cars,32)
            b = torch.sigmoid(self.actor3_b(state)) #(batch_size,num_cars,1)

            whether_select_probs = F.softmax(whether_select, dim=2) #(batch_size,num_cars,2)
            quant_probs = F.softmax(quant_select, dim=2)#(batch_size,num_cars,32)

            if self.training:
                quant_cate = torch.distributions.Categorical(quant_probs)
                select_cate = torch.distributions.Categorical(whether_select_probs)
                ptr_quant = quant_cate.sample()#(batch, num_cars)
                log_quant = quant_cate.log_prob(ptr_quant)#(batch, num_cars)
                ptr_select = select_cate.sample()##(batch, num_cars)
                logp_select = select_cate.log_prob(ptr_select)#(batch, num_cars)
            else:
                prob_quant, ptr_quant = torch.max(quant_probs, 1)  # Greedy
                log_quant = prob_quant.log()
                prob_select, ptr_select = torch.max(whether_select_probs, 1)  # Greedy
                logp_select = prob_select.log()


            #update
            dynamic_1,_ = torch.max(0.005*ptr_quant + 0.01*ptr_quant/b.squeeze(),dim=1)#(batch_size)
            dynamic_1 = dynamic_1.unsqueeze(1)#(batch_size,1)
            dynamic_2 = torch.mul(dynamic_1,static[:,:,2,step])+ dynamic[:, :, 1, step]
            dynamic_3 = dynamic_2 + 0.005 * ptr_quant
            dynamic_3 = torch.where(dynamic_3<500,500-dynamic_3,dynamic_3-500)
            dynamic[:, :, :,step+1] = torch.cat([dynamic_1.repeat(1,20).unsqueeze(2),
                                               dynamic_2.unsqueeze(2),
                                               dynamic_3.unsqueeze(2)],axis=2)
            # After visiting a node update the dynamic representation
            # if self.update_fn is not None:
            #     dynamic = self.update_fn(dynamic, ptr.data)
            #     dynamic_hidden = self.dynamic_encoder(dynamic)
            #
            #     # Since we compute the VRP in minibatches, some tours may have
            #     # number of stops. We force the vehicles to remain at the depot
            #     # in these cases, and logp := 0
            #     is_done = dynamic[:, 1].sum(1).eq(0).float()
            #     logp = logp * (1. - is_done)
            #
            # # And update the mask so we don't re-visit if we don't need to
            # if self.mask_fn is not None:
            #     mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            whether_select_lst.append(ptr_select.unsqueeze(2))
            quant_select_lst.append(ptr_select.unsqueeze(2))
            zero = torch.tensor(0)
            one = torch.tensor(1)
            b_lst.append(torch.where (b > 0.5, one, zero))
            whether_select_logp.append(logp_select.unsqueeze(2))
            quant_select_logp.append(log_quant.unsqueeze(2))
            b_logp.append(b)

        whether_select_seq = torch.cat(whether_select_lst,2).unsqueeze(2)
        quant_select_seq = torch.cat(quant_select_lst,2).unsqueeze(2)
        b_seq = torch.cat(b_lst,2).unsqueeze(2)
        whether_select_logp_seq = torch.cat(whether_select_logp,2).unsqueeze(2)
        quant_select_logp_seq = torch.cat(quant_select_logp,2).unsqueeze(2)
        b_logp_seq = torch.cat(b_logp,2).unsqueeze(2)
        action = torch.cat([whether_select_seq,quant_select_seq,b_seq], dim=2)  # (batch_size, num_cars,action_size, seq_len)
        action_logp = torch.cat([whether_select_logp_seq,quant_select_logp_seq,b_logp_seq], dim=2)   # (batch_size, num_cars,action_size, seq_len) # (batch_size, seq_len)

        return action, action_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
