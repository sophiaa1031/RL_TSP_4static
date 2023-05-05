"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
from agents.paddpg import Actor,Critic
import os
from tasks.flvn import RewardScaling
from tasks import flvn
from tasks.flvn import TSPDataset
import random
from plots import plot_reward
from plots import plot_time
from plots import plot_pf
from plots import plot_v_f1f2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')





# def update_fn(action,y):

def validate(data_loader, actor, reward_fn, w1, w2, obj1_scaling, obj2_scaling, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    rewards = []
    obj1s = []
    obj2s = []
    for batch_idx, batch in enumerate(data_loader):
        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            action, _ = actor(static, dynamic)

        reward, obj1, obj2 = reward_fn(static, dynamic, action, obj1_scaling, obj2_scaling, w1, w2)

        rewards.append(torch.mean(reward.detach()).item())
        obj1s.append(torch.mean(obj1.detach()).item())
        obj2s.append(torch.mean(obj2.detach()).item())
        # if render_fn is not None and batch_idx < num_plot:
        #     name = 'batch%d_%2.4f.png'%(batch_idx, torch.mean(reward.detach()).item())
        #     path = os.path.join(save_dir, name)
        #     render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards), np.mean(obj1s), np.mean(obj2s)


def train(actor, critic, w1, w2, v_min, task, num_cars, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, obj1_scaling, obj2_scaling, epoch,change_in_every_epoch,episode,iteration,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    bname = "_transfer"
    save_dir = os.path.join(task + bname, '%d_num_cars' % num_cars, 'w_%2.2f_%2.2f' % (w1, w2), now)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf
    start_total = time.time()
    valid_reward = []
    train_loss = []
    train_reward = []

    for epoch in range(epoch):
        print("epoch %d start:" % epoch)
        actor.train()  # model train -> dropout   training ->dropout 随机丢弃掉一些神经元0.3    testing  dropout 值*0.3
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []  # 缓存中的奖励
        obj1s, obj2s = [], []

        epoch_start = time.time()
        start = epoch_start

        if change_in_every_epoch == True:
            train_data = TSPDataset(episode, num_cars, iteration, random.randint(0,100000),v_min)
            train_loader = DataLoader(train_data, batch_size, True, num_workers=0)

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # 根据当前状态(state)从Actor中输出动作的概率分布
            action, action_logp = actor(static, dynamic)  # actor.forward(static, dynamic, x0)

            # 执行动作,计算奖励
            reward, obj1, obj2 = reward_fn(static, dynamic, action, obj1_scaling, obj2_scaling, w1, w2)

            # Critic评估状态价值
            critic_est = critic(static[:, :, :, :-1], dynamic[:, :, :, :-1],
                                ).view(-1)

            # 计算优势函数
            advantage = (reward.view(-1) - critic_est)

            # Actor参数更新
            # actor_loss1 = torch.mean(advantage.detach() * action_logp[:,:,0,:].sum(dim=1).view(-1))  # 计算Actor的损失函数
            actor_loss2 = torch.mean(advantage.detach() * action_logp[:, :, 1, :].sum(dim=1).view(-1))  # 计算Actor的损失函数
            actor_loss3 = torch.mean(advantage.detach() * action_logp[:, :, 2, :].sum(dim=1).view(-1))  # 计算Actor的损失函数
            actor_loss = actor_loss2 + actor_loss3
            if torch.isnan(actor_loss).any().item() or torch.isinf(actor_loss).any().item():
                print('something wrong 1')
            actor_optim.zero_grad()
            actor_loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()  # 更新模型参数

            # Critic参数更新
            critic_loss = torch.mean(advantage ** 2)  # 计算Critic的损失函数
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            # 存入relay buffer
            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())
            obj1s.append(torch.mean(obj1.detach()).item())
            obj2s.append(torch.mean(obj2.detach()).item())

            if (batch_idx + 1) % 1 == 0:
                # print("\n")
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])
                mean_obj1 = np.mean(obj1s[-100:])
                mean_obj2 = np.mean(obj2s[-100:])
                print('  Batch %d/%d, reward: %2.3f, obj1: %2.3f, obj2: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_obj1, mean_obj2, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        train_loss.append(mean_loss)
        train_reward.append(mean_reward)
        mean_valid, mean_obj1_valid, mean_obj2_valid = validate(valid_loader, actor, reward_fn, w1, w2, obj1_scaling,
                                                                obj2_scaling, render_fn,
                                                                '.', num_plot=5)
        valid_reward.append(mean_valid)
        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            # 存在w_1_0主文件夹下，多存一份，用来transfer to next w
            main_dir = os.path.join(task + bname, '%d_num_cars' % num_cars, 'w_%2.2f_%2.2f' % (w1, w2))
            save_path = os.path.join(main_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)
            save_path = os.path.join(main_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward/valid: %2.4f, %2.4f, %2.4f, obj1_valid: %2.3f, obj2_valid: %2.3f. took: %2.4fs ' \
              '(%2.4fs / 100 batches)' % \
              (mean_loss, mean_reward, mean_valid, mean_obj1_valid, mean_obj2_valid, time.time() - epoch_start,
               np.mean(times)))

    total_time = time.time() - start_total
    print("Total run time of epoches: %2.4f" % (total_time))
    # 存time画图
    f = open('figure/time.txt', 'a')
    f.write('{:.4f}'.format(total_time)+',')
    return train_loss, train_reward, valid_reward

def train_tsp(args, w1=1, w2=0, checkpoint=None,v_min=8):

    train_data = TSPDataset(args.episode, args.num_cars, args.iteration, args.seed,v_min)
    valid_data = TSPDataset(args.episode, args.num_cars, args.iteration, args.seed + 1,v_min)
    test_data = TSPDataset(args.episode, args.num_cars, args.iteration, args.seed + 2,v_min)

    update_fn = None

    obj1_scaling = RewardScaling()
    obj2_scaling = RewardScaling()

    actor = Actor(args.static_size,
                  args.dynamic_size,
                  args.action_size,
                  args.action_parameter_size,
                  args.hidden_size,
                  update_fn,
                  args.iteration,
                  args.num_cars).to(device)  # 定义一个Actor模型

    critic = Critic(args.static_size,
                         args.dynamic_size,
                         args.action_size,
                         args.action_parameter_size,
                         args.hidden_size,
                         args.num_cars).to(device)  # 定义一个Critic模型

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = flvn.reward
    kwargs['render_fn'] = flvn.render
    kwargs['obj1_scaling'] = obj1_scaling
    kwargs['obj2_scaling'] = obj2_scaling
    # kwargs['epoch'] = args.epoch
    # kwargs['change_in_every_epoch'] = args.change_in_every_epoch
    # kwargs['episode'] = args.episode
    # kwargs['iteration'] = args.iteration

    # parameter transfer
    if checkpoint:
        path = os.path.join(checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))
        # actor.static_encoder.state_dict().get("conv.weight").size()
        path = os.path.join(checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))
        print('Now loading the weight in ', checkpoint)

    if not args.test:
        train_loss, train_reward, valid_reward = train(actor, critic, w1, w2, v_min,**kwargs)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.episode, False, num_workers=0)
    out = validate(test_loader, actor, flvn.reward, w1, w2, obj1_scaling, obj2_scaling, flvn.render, test_dir,
                   num_plot=5)

    print('w1=%2.2f,w2=%2.2f. Average objectives: ' % (w1, w2), out)
    print("")

    # 存reward画图
    f1 = open('figure/reward/{:.2f}_{:.2f}.txt'.format(w1,w2),'w')
    f1.write('train_loss:')
    f1.write(','.join(map(str,train_loss)))
    f1.write('\ntrain_reward:')
    f1.write(','.join(map(str,train_reward)))
    f1.write('\nvalid_reward:')
    f1.write(','.join(map(str,valid_reward)))
    print('Saving the weight', w1, w2)

    # 存optimal value画图
    f2 = open('figure/optimalvalue.txt', 'a')
    f2.write('{:.4f}'.format(out[1])+',')
    f2.write('{:.4f}\n'.format(out[2]))

def train_tsp_final(args, w1=0.5, w2=0.5, checkpoint=None,v_min=8):

    train_data = TSPDataset(args.episode, args.num_cars, args.iteration, args.seed,v_min)
    valid_data = TSPDataset(args.episode, args.num_cars, args.iteration, args.seed + 1,v_min)
    test_data = TSPDataset(args.episode, args.num_cars, args.iteration, args.seed + 2,v_min)

    update_fn = None

    obj1_scaling = RewardScaling()
    obj2_scaling = RewardScaling()

    actor = Actor(args.static_size,
                  args.dynamic_size,
                  args.action_size,
                  args.action_parameter_size,
                  args.hidden_size,
                  update_fn,
                  flvn.update_mask,
                  args.num_layers,
                  args.dropout,
                  args.iteration,
                  args.num_cars).to(device)  # 定义一个Actor模型

    critic = Critic(args.static_size,
                         args.dynamic_size,
                         args.action_size,
                         args.action_parameter_size,
                         args.hidden_size,
                         args.num_cars).to(device)  # 定义一个Critic模型

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = flvn.reward
    kwargs['render_fn'] = flvn.render
    kwargs['obj1_scaling'] = obj1_scaling
    kwargs['obj2_scaling'] = obj2_scaling
    # kwargs['epoch'] = args.epoch
    # kwargs['change_in_every_epoch'] = args.change_in_every_epoch
    # kwargs['episode'] = args.episode
    # kwargs['iteration'] = args.iteration

    # parameter transfer
    if checkpoint:
        path = os.path.join(checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))
        # actor.static_encoder.state_dict().get("conv.weight").size()
        path = os.path.join(checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))
        print('Now loading the weight in ', checkpoint)

    if not args.test:
        train_loss, train_reward, valid_reward = train(actor, critic, w1, w2, v_min,**kwargs)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.episode, False, num_workers=0)
    out = validate(test_loader, actor, flvn.reward, w1, w2, obj1_scaling, obj2_scaling, flvn.render, test_dir,
                   num_plot=5)

    print('w1=%2.2f,w2=%2.2f. Average objectives: ' % (w1, w2), out)
    print("")

    # 存reward画图
    f1 = open('figure/reward/finak_test.txt','w')
    f1.write('train_loss:')
    f1.write(','.join(map(str,train_loss)))
    f1.write('\ntrain_reward:')
    f1.write(','.join(map(str,train_reward)))
    f1.write('\nvalid_reward:')
    f1.write(','.join(map(str,valid_reward)))

    # 存optimal value画图
    f2 = open('figure/optimalvalue.txt', 'a')
    f2.write('{:.4f}'.format(out[1])+',')
    f2.write('{:.4f}\n'.format(out[2]))

    # 存不同v的时候optimal value画图
    f3 = open('figure/optimalvalue_v.txt', 'a')
    f3.write('{:.4f}'.format(out[1]) + ',')
    f3.write('{:.4f}\n'.format(out[2]))

if __name__ == '__main__':
    # num_cars = 10_num_cars
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    # parser.add_argument('--checkpoint', default="tsp/20/w_1_0/20_06_30.888074")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tsp')
    # parser.add_argument('--nodes', dest='num_cars', default=num_cars, type=int)
    parser.add_argument('--num_cars', default=10, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=1024*8, type=int)  # GPU上限1024*8
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--episode', default=10000, type=int)
    parser.add_argument('--iteration', default=20, type=int)
    parser.add_argument('--static_size', default=4, type=int)
    parser.add_argument('--dynamic_size', default=3, type=int)
    parser.add_argument('--action_size', default=8, type=int)
    parser.add_argument('--action_parameter_size', default=8, type=int)
    parser.add_argument('--subproblem_size', default=5, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--change_in_every_epoch', default=True, type=bool)


    args = parser.parse_args()
    v_min_list = [8,13,18,23,28] #[8,13,18,23,28]

    f = open('figure/optimalvalue_v.txt', 'w')
    start_all = time.time()
    for v_min in  v_min_list:
        w2_list = 1 - np.arange(args.subproblem_size + 1) / args.subproblem_size
        f = open('figure/time.txt', 'w')
        f = open('figure/optimalvalue.txt', 'w')
        directory = 'figure/reward'
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
        for i in range(0, args.subproblem_size + 1):
            print("Current v:%2.0f,w:%2.2f/%2.2f" % (v_min, 1 - w2_list[i], w2_list[i]))
            if i == 0:
                # The first subproblem can be trained from scratch. It also can be trained based on a
                # single-TSP trained model, where the model can be obtained from everywhere in github
                checkpoint = None
            else:
                # Parameter transfer. train based on the parameters of the previous subproblem
                checkpoint = 'tsp_transfer/%d_num_cars/w_%2.2f_%2.2f' % (args.num_cars, 1 - w2_list[i - 1], w2_list[i - 1])
            train_tsp(args, 1 - w2_list[i], w2_list[i], checkpoint,v_min)
        # 测试（0.5，0.5）
        train_tsp_final(args, 0.5, 0.5, 'tsp_transfer/%d_num_cars/w_%2.2f_%2.2f' % (args.num_cars, 1 - w2_list[-1], w2_list[-1]),v_min)
        # train_tsp(args, 0.5, 0.5, None)
        # plot_reward()
        # plot_time()
        # plot_pf()
    end_all = time.time()
    print('The overall time is:%2.2f s' % (end_all-start_all))
    # 画图
    plot_v_f1f2()

