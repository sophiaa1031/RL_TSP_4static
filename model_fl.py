import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

class Actor(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0., iteration=20, num_cars=10):
        super(Actor, self).__init__()
        self.update_fn = update_fn
        self.mask_fn = update_fn
        # Define the encoder & decoder models
        self.static_encoder = nn.Linear(static_size, hidden_size)
        self.dynamic_encoder = nn.Linear(dynamic_size, hidden_size)
        self.static_bn = torch.nn.BatchNorm1d(static_size, eps=1e-05)
        self.dynamic_bn = torch.nn.BatchNorm1d(dynamic_size, eps=1e-05) # 对输入张量进行归一化操作
        # self.actor1_select = nn.Linear(hidden_size * 2, 2)
        self.actor2_quant = nn.Linear(hidden_size * 2, 32)
        self.actor3_b = nn.Linear(hidden_size * 2, 2)
        self.iteration = iteration
        self.num_cars = num_cars

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # max_steps = self.iteration  # if self.mask_fn is None else 50

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.

        whether_select_lst, quant_select_lst, b_lst = [], [], []
        whether_select_logp, quant_select_logp, b_logp = [], [], []


        # batch_size, num_cars, features, steps = static.shape
        # static = static.view(batch_size * num_cars, features, steps)
        # static = self.static_bn(static.unsqueeze(2))  # Add a singleton dimension for 'height' and apply BN
        # static = static.squeeze(2)  # Remove the singleton dimension for 'height'
        # static = static.view(batch_size, num_cars, features, steps)
        #
        # batch_size, num_cars, features, steps = dynamic.shape
        # dynamic = dynamic.view(batch_size * num_cars, features,
        #                      steps)  # Reshape to [batch_size*num_cars, features, steps]
        # # BatchNorm2d along feature dimension
        # dynamic = self.dynamic_bn(dynamic.unsqueeze(2))  # Add a singleton dimension for 'height' and apply BN
        # dynamic = dynamic.squeeze(2)  # Remove the singleton dimension for 'height'
        # dynamic = dynamic.view(batch_size, num_cars, features, steps)

        for step in range(self.iteration):
            # 更新state网络层
            static_temp = static[:, :, :, step]
            batch_size, num_cars, features = static_temp.shape
            static_temp = static_temp.view(batch_size * num_cars, features)
            static_temp = self.static_bn(static_temp)  # Add a singleton dimension for 'height' and apply BN
            static_temp = static_temp.view(batch_size, num_cars, features)
            static_hidden = self.static_encoder(static_temp)  # []
            dynamic_temp = dynamic[:, :, :, step].clone()
            batch_size, num_cars, features = dynamic_temp.shape
            dynamic_temp = dynamic_temp.view(batch_size * num_cars, features)  # Reshape to [batch_size*num_cars, features, steps]
            # BatchNorm2d along feature dimension
            dynamic_temp = self.dynamic_bn(dynamic_temp)  # Add a singleton dimension for 'height' and apply BN
            dynamic_temp = dynamic_temp.view(batch_size, num_cars, features)
            dynamic_hidden = self.dynamic_encoder(dynamic_temp)
            state = torch.cat((static_hidden, dynamic_hidden), 2)  # [batch_size,num_cars,2*hidden_size]

            # 根据state输出action
            # whether_select = self.actor1_select(state)  # (batch_size,num_cars,2)
            whether_select = torch.ones(batch_size,num_cars,2).to(device)
            quant_select = self.actor2_quant(state)  # (batch_size,num_cars,32)
            bdw_beforenorm = F.softmax(self.actor3_b(state), dim=2)[:,:,1]
            bdw = bdw_beforenorm/(torch.sum(bdw_beforenorm,dim=1).unsqueeze(1).repeat(1, num_cars))  # (batch_size,num_cars,1)
            whether_select_probs = F.softmax(whether_select, dim=2)  # (batch_size,num_cars,2)
            quant_probs = F.softmax(quant_select, dim=2)  # (batch_size,num_cars,32)

            # 根据概率分布选择一个动作action
            if self.training:
                quant_cate = torch.distributions.Categorical(quant_probs)
                select_cate = torch.distributions.Categorical(whether_select_probs)
                ptr_quant = quant_cate.sample()  # (batch, num_cars)
                log_quant = quant_cate.log_prob(ptr_quant)  # (batch, num_cars)
                ptr_select = select_cate.sample()  # (batch, num_cars)
                logp_select = select_cate.log_prob(ptr_select)  # (batch, num_cars)
            else:
                prob_quant, ptr_quant = torch.max(quant_probs, 2)  # Greedy
                log_quant = prob_quant.log()
                prob_select, ptr_select = torch.max(whether_select_probs, 2)  # Greedy
                logp_select = prob_select.log()

            # 更新动态state (dynamic)
            with torch.no_grad():
                ptr_quant = ptr_quant + 1
                # save the maximum latency in the last iteration
                rate = bdw.detach() * 10 * torch.log2(1+1e7 * static[:, :, 0, step].clone() * torch.pow(dynamic[:, :, 2, step].clone(), -2))
                dynamic_0, _ = torch.max(0.002 * ptr_quant.clone()/static[:,:,1,step].clone() + ptr_quant.clone()/32 / rate, dim=1)  # (batch_size)
                # dynamic_0, _ = torch.max(0.002 * ptr_select.clone() * ptr_quant.clone() / static[:, :, 1,step].clone() + ptr_select.clone() * ptr_quant.clone() /
                #    (320 * bdw.detach() * torch.log2(1 + (1e7 * static[:, :, 0, step].clone() / (dynamic[:, :, 2, step].clone() * dynamic[:, :, 2,step].clone())))),dim=1)  # (batch_size)
                dynamic_0 = dynamic_0.unsqueeze(1)  # (batch_size,1)
                # update the current travel distance
                dynamic_1 = torch.mul(dynamic_0, static[:, :, 2, step].clone()) + dynamic[:, :, 1, step].clone()
                # update the distance from the RSU
                dynamic_2 = dynamic_1 + 0.005 * ptr_quant
                dynamic_2 = torch.sqrt(torch.pow(dynamic_2 - 300,2)+10**2)
                if torch.any(dynamic_2>301):
                    print('1')
                dynamic[:, :, :, step + 1] = torch.cat([dynamic_0.repeat(1, num_cars).unsqueeze(2),
                                                        dynamic_1.unsqueeze(2),
                                                        dynamic_2.unsqueeze(2)], axis=2).clone()
                if torch.isnan(dynamic).any().item() or torch.isinf(dynamic).any().item():
                    print('1')

            # 记录当前动作
            whether_select_lst.append(ptr_select.unsqueeze(2))
            quant_select_lst.append(ptr_quant.unsqueeze(2))
            b_lst.append(bdw.unsqueeze(2))
            whether_select_logp.append(logp_select.unsqueeze(2))
            quant_select_logp.append(log_quant.unsqueeze(2))
            b_logp.append(bdw.unsqueeze(2))

        # 更新动作action空间
        whether_select_seq = torch.cat(whether_select_lst, 2).unsqueeze(2)
        quant_select_seq = torch.cat(quant_select_lst, 2).unsqueeze(2)
        b_seq = torch.cat(b_lst, 2).unsqueeze(2).detach()
        action = torch.cat([whether_select_seq, quant_select_seq, b_seq],
                           dim=2)  # (batch_size, num_cars,action_size, seq_len)

        whether_select_logp_seq = torch.cat(whether_select_logp, 2).unsqueeze(2)
        quant_select_logp_seq = torch.cat(quant_select_logp, 2).unsqueeze(2)
        b_logp_seq = torch.cat(b_logp, 2).unsqueeze(2)
        action_logp = torch.cat([whether_select_logp_seq, quant_select_logp_seq, b_logp_seq],
                                dim=2)  # (batch_size, num_cars,action_size, seq_len) # (batch_size, seq_len)

        return action, action_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
