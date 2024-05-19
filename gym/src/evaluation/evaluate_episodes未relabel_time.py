import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        z_dim,
        h_model,
        l_model,
        horizon,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    h_model.eval()
    h_model.to(device=device)
    l_model.eval()
    l_model.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]
    for i in range(episodes_times):
        state = env.reset()
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        if i == 0:
            states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            z_distributions = torch.zeros((0, 2 * z_dim), device=device, dtype=torch.float32)
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        else:
            refresh_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            refresh_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            refresh_timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            states = torch.cat([states, refresh_state], dim=0)
            target_return = torch.cat([target_return, refresh_return], dim=1)
            timesteps = torch.cat([timesteps,refresh_timesteps], dim=1)
        t, episode_return, episode_length = 0, 0, 0
        while t < max_ep_len:
            # add padding
            z_distributions = torch.cat([z_distributions, torch.zeros((1, 2*z_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            z_distribution_predict = h_model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                z_distributions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            z_distributions[-1] = z_distribution_predict
            tmp_return = target_return[0,-1]
            for i in range(horizon):
                actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                action = l_model.get_action(state,z_distribution_predict)
                actions[-1] = action.reshape(-1,act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                t+=1
                cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                rewards[-1] = reward
                if mode != 'delayed':
                    tmp_return = tmp_return - (reward/scale)
                else:
                    tmp_return = target_return[0,-1]
                episode_return += reward
                episode_length += 1
                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    break
            if done:
                break
            states = torch.cat([states, cur_state], dim=0)
            target_return = torch.cat(
                [target_return, tmp_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths


def evaluate_episode_rtg2(
        env,
        state_dim,
        act_dim,
        z_dim,
        h_model,
        l_model,
        horizon,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    h_model.eval()
    h_model.to(device=device)
    l_model.eval()
    l_model.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]
    for episodes_time in range(episodes_times):
        state = env.reset()
        state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        if episodes_time == 0:
            # states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = state.reshape(1, state_dim)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            z_distributions = torch.zeros((0, 2 * z_dim), device=device, dtype=torch.float32)
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        else:
            # refresh_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            refresh_state = state.reshape(1, state_dim)
            refresh_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            refresh_timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            states = torch.cat([states, refresh_state], dim=0)
            target_return = torch.cat([target_return, refresh_return], dim=1)
            timesteps = torch.cat([timesteps,refresh_timesteps], dim=1)
        t, episode_return, episode_length = 0, 0, 0
        while t < max_ep_len:
            # add padding
            z_distributions = torch.cat([z_distributions, torch.zeros((1, 2*z_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            z_distribution_predict = h_model.get_action(
                # (states.to(dtype=torch.float32) - state_mean) / state_std,
                states.to(dtype=torch.float32),
                z_distributions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            tmp_return = target_return[0,-1]
            l_states = states[-1].reshape(1,-1)
            l_time_steps = timesteps[0,-1].reshape(-1,1)
            l_actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            for i in range(horizon):
                l_actions = torch.cat([l_actions, torch.zeros((1, act_dim), device=device)], dim=0)
                action = l_model.get_action(state, z_distribution_predict)
                l_actions[-1] = action.reshape(-1, act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
                t+=1
                # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                cur_state = state.reshape(1, state_dim)
                rewards[-1] = reward
                if mode != 'delayed':
                    tmp_return = tmp_return - (reward/scale)
                else:
                    tmp_return = target_return[0,-1]
                episode_return += reward
                episode_length += 1
                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    break
                if i+1 < horizon:
                    l_states = torch.cat([l_states, cur_state], dim=0)
                    l_time_steps = torch.cat([l_time_steps, torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
            z_distributions[-1] = z_distribution_predict
            # z_actual_distributions = l_model.get_actual_distribution(l_states,l_actions,l_time_steps)
            # z_distributions[-1] = z_actual_distributions
            if done:
                break
            #TODO episode结束位置的插入操作和训练部分是否一致
            states = torch.cat([states, cur_state], dim=0)
            target_return = torch.cat(
                [target_return, tmp_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths


def evaluate_episode_rtg3(
        env,
        state_dim,
        act_dim,
        z_dim,
        h_model,
        l_model,
        horizon,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    h_model.eval()
    h_model.to(device=device)
    l_model.eval()
    l_model.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]
    for episodes_time in range(episodes_times):
        state = env.reset()
        state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        if episodes_time == 0:
            # states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = state.reshape(1, state_dim)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            z_distributions = torch.zeros((0, 2 * z_dim), device=device, dtype=torch.float32)
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        else:
            # refresh_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            refresh_state = state.reshape(1, state_dim)
            refresh_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            refresh_timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            states = torch.cat([states, refresh_state], dim=0)
            target_return = torch.cat([target_return, refresh_return], dim=1)
            timesteps = torch.cat([timesteps,refresh_timesteps], dim=1)
        t, episode_return, episode_length = 0, 0, 0
        while t < max_ep_len:
            # add padding
            z_distributions = torch.cat([z_distributions, torch.zeros((1, 2*z_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            z_distribution_predict = h_model.get_action(
                # (states.to(dtype=torch.float32) - state_mean) / state_std,
                states.to(dtype=torch.float32),
                z_distributions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            tmp_return = target_return[0,-1]
            l_states = states[-1].reshape(1,-1)
            l_time_steps = timesteps[0,-1].reshape(-1,1)
            l_actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            for i in range(horizon):
                l_actions = torch.cat([l_actions, torch.zeros((1, act_dim), device=device)], dim=0)
                action = l_model.get_action(state, z_distribution_predict)
                l_actions[-1] = action.reshape(-1, act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
                t+=1
                # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                cur_state = state.reshape(1, state_dim)
                rewards[-1] = reward
                if mode != 'delayed':
                    tmp_return = tmp_return - (reward/scale)
                else:
                    tmp_return = target_return[0,-1]
                episode_return += reward
                episode_length += 1
                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    break
                if i+1 < horizon:
                    l_states = torch.cat([l_states, cur_state], dim=0)
                    l_time_steps = torch.cat([l_time_steps, torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
            # z_distributions[-1] = z_distribution_predict
            z_actual_distributions = l_model.get_actual_distribution(l_states,l_actions,l_time_steps)
            z_distributions[-1] = z_actual_distributions
            if done:
                break
            #TODO episode结束位置的插入操作和训练部分是否一致
            states = torch.cat([states, cur_state], dim=0)
            target_return = torch.cat(
                [target_return, tmp_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths
