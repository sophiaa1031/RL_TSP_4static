import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

class Actor(nn.Module):
    def __init__(self, static_size, dynamic_size, action_size, action_parameter_size, hidden_size,
                 update_fn=None, iteration=20, num_cars=10):
        super(Actor, self).__init__()
        self.update_fn = update_fn
        self.mask_fn = update_fn
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        # Define the encoder & decoder models
        self.static_encoder = nn.Linear(static_size, hidden_size)
        self.dynamic_encoder = nn.Linear(dynamic_size, hidden_size)
        self.static_bn = torch.nn.BatchNorm1d(static_size, eps=1e-05)
        self.dynamic_bn = torch.nn.BatchNorm1d(dynamic_size, eps=1e-05) # 对输入张量进行归一化操作
        # self.actor1_select = nn.Linear(hidden_size * 2, 2)
        # self.actor2_quant = nn.Linear(hidden_size * 2, 9)
        # self.actor3_b = nn.Linear(hidden_size * 2, 2)

        self.iteration = iteration
        self.num_cars = num_cars


        self.action_output_layer = nn.Linear(hidden_size * 2, self.action_size)
        self.action_parameters_output_layer = nn.Linear(hidden_size * 2, self.action_parameter_size)

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

        for step in range(self.iteration):
            # 更新state网络层
            static_temp = static[:, :, :, step]
            batch_size, num_cars, features = static_temp.shape
            t_comp = (torch.ones(batch_size, num_cars) * 0.05).to(device)
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

            actions = self.action_output_layer(state)
            action_params = self.action_parameters_output_layer(state)
            ##################
            # 优化带宽b/随机生成b
            ##################
            # bdw_beforenorm = torch.rand(batch_size,num_cars).to(device)

            whether_select = torch.ones(batch_size, num_cars, 2).to(device)
            whether_select_probs = F.softmax(whether_select, dim=2)  # (batch_size,num_cars,2)
            quant_probs = F.softmax(actions, dim=2)  # (batch_size,num_cars,32)


            prob_quant, ptr_quant = torch.max(quant_probs, 2)  # Greedy
            log_quant = prob_quant.log()
            prob_select, ptr_select = torch.max(whether_select_probs, 2)  # Greedy
            logp_select = prob_select.log()
            bdw = torch.gather(action_params, 2, ptr_quant.unsqueeze(-1)).squeeze(-1)
            bdw = F.softmax(bdw, dim=1)
            # 更新动态state (dynamic)
            with torch.no_grad():
                ptr_quant = ptr_quant + 8 # 量化程度为[8,16]
                # update the distance from the RSU, add computation time
                dynamic_2 = dynamic[:, :, 1, step] + torch.mul(t_comp, static[:, :, 2, step].clone())
                dynamic_2 = torch.sqrt(torch.pow(dynamic_2 - 500, 2) + 10 ** 2)
                # save the maximum latency in the last iteration
                rate = bdw.detach() * 10 * torch.log2(1+1e7 * static[:, :, 0, step].clone() * torch.pow(dynamic_2, -2))
                dynamic_0, _ = torch.max(t_comp + ptr_quant.clone()/32 / rate, dim=1)  # (batch_size)
                dynamic_0 = dynamic_0.unsqueeze(1)  # (batch_size,1)
                # update the current travel distance
                dynamic_1 = torch.mul(dynamic_0, static[:, :, 2, step].clone()) + dynamic[:, :, 1, step].clone()
                # if torch.any(dynamic_2>501):
                #     print('something wrong')
                dynamic[:, :, :, step + 1] = torch.cat([dynamic_0.repeat(1, num_cars).unsqueeze(2),
                                                        dynamic_1.unsqueeze(2),
                                                        dynamic_2.unsqueeze(2)], axis=2).clone()
                # if torch.isnan(dynamic).any().item() or torch.isinf(dynamic).any().item():
                #     print('something wrong 1')
            # 记录当前动作
            whether_select_lst.append(ptr_select.unsqueeze(2))
            quant_select_lst.append(ptr_quant.unsqueeze(2))
            b_lst.append(bdw.unsqueeze(2))
            whether_select_logp.append(whether_select_probs.unsqueeze(3))
            quant_select_logp.append(quant_probs.unsqueeze(3))
            b_logp.append(action_params.unsqueeze(3))

        # 更新动作action空间
        whether_select_seq = torch.cat(whether_select_lst, 2).unsqueeze(2)
        quant_select_seq = torch.cat(quant_select_lst, 2).unsqueeze(2)
        b_seq = torch.cat(b_lst, 2).unsqueeze(2).detach()
        action = torch.cat([whether_select_seq, quant_select_seq, b_seq],
                           dim=2)  # (batch_size, num_cars,action_size, seq_len)

        whether_select_logp_seq = torch.cat(whether_select_logp, 3)# [batch_size, cars,action_size, iteration]
        quant_select_logp_seq = torch.cat(quant_select_logp, 3)
        b_logp_seq = torch.cat(b_logp, 3)
        action_logp = torch.cat([whether_select_logp_seq, quant_select_logp_seq, b_logp_seq],
                                dim=2)  # (batch_size, num_cars,action_size, seq_len) # (batch_size, seq_len)

        return action, action_logp

class Critic(nn.Module):
    # 根据输入的状态输出状态的价值
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, action_size,
                 action_parameter_size,
                 hidden_size, num_cars):
        super(Critic, self).__init__()

        self.static_encoder = nn.Linear(static_size * num_cars, hidden_size)
        self.dynamic_encoder = nn.Linear(dynamic_size * num_cars, hidden_size)
        self.action_encoder = nn.Linear((action_size + action_parameter_size) * num_cars, hidden_size)
        self.static_bn = torch.nn.BatchNorm2d(static_size, eps=1e-05)
        self.dynamic_bn = torch.nn.BatchNorm2d(dynamic_size, eps=1e-05)
        # Define the encoder & decoder models
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, actions, action_parameters):
        batch_size, num_cars, features, steps = static.shape
        static = static.view(batch_size * num_cars, features, steps)
        static = self.static_bn(static.unsqueeze(2))  # Add a singleton dimension for 'height' and apply BN
        static = static.squeeze(2)  # Remove the singleton dimension for 'height'
        static = static.view(batch_size, num_cars * features, steps)
        static_hidden = self.static_encoder(static.transpose(2, 1))

        batch_size, num_cars, features, steps = dynamic.shape
        dynamic = dynamic.view(batch_size * num_cars, features,
                               steps)  # Reshape to [batch_size*num_cars, features, steps]
        # BatchNorm2d along feature dimension
        dynamic = self.dynamic_bn(dynamic.unsqueeze(2))  # Add a singleton dimension for 'height' and apply BN
        dynamic = dynamic.squeeze(2)  # Remove the singleton dimension for 'height'
        dynamic = dynamic.view(batch_size, num_cars * features, steps)
        dynamic_hidden = self.dynamic_encoder(dynamic.transpose(2, 1))
        action_all = torch.cat((actions, action_parameters), 2)
        action_hidden = self.action_encoder(action_all)
        hidden = torch.cat((static_hidden, dynamic_hidden,action_hidden), 2)


        output = F.relu(self.fc1(hidden))
        output = self.fc2(output).squeeze()
        return output


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
