import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class DRL4TSP(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()
        self.update_fn = update_fn
        self.mask_fn = update_fn
        # Define the encoder & decoder models
        self.static_encoder = nn.Linear(static_size, hidden_size)
        self.dynamic_encoder = nn.Linear(dynamic_size, hidden_size)
        self.actor1_select = nn.Linear(hidden_size*2, 2)
        self.actor2_quant = nn.Linear(hidden_size*2, 32)
        self.actor3_b = nn.Linear(hidden_size*2, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        batch_size, input_size, sequence_size = static.size()   #   sequence_size = 50

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 50

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        print ("static.shape",static.shape)
        static_hidden = self.static_encoder(static.view(batch_size,-1))
        dynamic_hidden = self.dynamic_encoder(dynamic.view(batch_size,-1))
        state = torch.cat((static_hidden, dynamic_hidden), 1)

        whether_select_lst, quant_select_lst, b_lst = [], [], []
        whether_select_logp, quant_select_logp, b_logp = [], [], []
        for _ in range(max_steps):

            if not mask.byte().any():
                break


            whether_select = self.actor1_select(state) # (batch_)
            quant_select = self.actor2_quant(state)
            b = self.actor3_b(state)

            whether_select_probs = F.softmax(whether_select, dim=1)
            quant_probs = F.softmax(quant_select, dim=1)

            if self.training:
                quant_cate = torch.distributions.Categorical(quant_probs)
                select_cate = torch.distributions.Categorical(whether_select_probs)
                ptr_quant = quant_cate.sample()
                log_quant = quant_cate.log_prob(ptr_quant)
                ptr_select = select_cate.sample()
                logp_select = select_cate.log_prob(ptr_select)
            else:
                prob_quant, ptr_quant = torch.max(quant_probs, 1)  # Greedy
                log_quant = prob_quant.log()
                prob_select, ptr_select = torch.max(whether_select_probs, 1)  # Greedy
                logp_select = prob_select.log()

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

            whether_select_lst.append(ptr_select.unsqueeze(1))
            quant_select_lst.append(ptr_select.unsqueeze(1))
            zero = torch.tensor(0)
            one = torch.tensor(1)
            b_lst.append(torch.where (b > 0.5, one, zero))
            whether_select_logp.append(logp_select.unsqueeze(1))
            quant_select_logp.append(log_quant.unsqueeze(1))
            b_logp.append(b)


        action = torch.cat([whether_select_lst,quant_select_lst,b_lst], dim=1)  # (batch_size, seq_len)
        action_logp = torch.cat([whether_select_logp,quant_select_logp,b_logp], dim=1)  # (batch_size, seq_len)

        return action, action_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
