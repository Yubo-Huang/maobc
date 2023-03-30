# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation in the offline mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

from absl import logging


import gin
import numpy as np
import tensorflow.compat.v1 as tf
import dataset
import train_eval_utils
import utils
from utils import read_csv_file, merge_transitions, numpy2tensor
from space import state_space, action_space
from group import group_set

Transition = dataset.Transition

@gin.configurable
def train_eval_offline(
    # Basic args.
    log_dir,
    data_dir,
    agent_module,
    num_agents,
    env_name='Sofwa9',
    n_train=int(1e6),
    shuffle_steps=0,
    seed=0,
    use_seed_for_data=False,
    # Train and eval args.
    total_train_steps=int(1e5),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(2000),
    # Agent args.
    model_params=(((64, 64),), 2),
    # model_params=(((64, 64),), 2, 0.05),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99
    ):
  """Training a policy with a fixed dataset."""
  # Create tf_env to get specs.
  num_state = 24
  num_action = 2
  observation_spec = state_space(shape=(num_state,), maximum=100*np.ones(num_state), minimum=-100*np.ones(num_state))
  # actions: CT prime (0.1, 1), yaw angle (-30, 30)
  action_spec = action_space(shape=(num_action,), maximum=np.array([1.1, 30]), minimum=np.array([0.1, -30]))
  group_state_spec = state_space(shape=(num_state*num_agents,), maximum=100*np.ones(num_state*num_agents), minimum=-100*np.ones(num_state*num_agents))
  group_action_spec = action_space(shape=(num_action*num_agents,), maximum=np.array([1.1, 30]), minimum=np.array([0.1, -30]))
  # prepare data
  data_size = int(1e6)
  group_samples = {'s1': [], 's2': [], 'a1': [],  'a2': [], 'discount': [], 'reward': []}
  group_dataset = dataset.Dataset(group_state_spec, group_action_spec, data_size)
  agent_samples = [[] for _ in range(num_agents)]
    
  dirs = os.listdir(data_dir)
  for dir in dirs:
    dir_path = os.path.join(data_dir, dir)
    files = os.listdir(dir_path)
    files.sort()
    print('including the dataset:', dir)
    for i, file in enumerate(files):
      print('including the data file:', file)
      file_path = os.path.join(dir_path, file)
      isample = read_csv_file(file_path, num_state, num_action)
      if agent_samples[i] == []:
        agent_samples[i] = isample
      else:
        agent_samples[i] = merge_transitions(agent_samples[i], isample)    
        
  agent_dataset = []
  for isample in agent_samples:
    group_samples['s1'].append(isample.s1)
    group_samples['s2'].append(isample.s2)
    group_samples['a1'].append(isample.a1)
    group_samples['a2'].append(isample.a2)
    group_samples['reward'].append(isample.reward)
    group_samples['discount'].append(isample.discount)
    agent_dataset.append(dataset.Dataset(observation_spec, action_spec, data_size))
    isample = numpy2tensor(isample)
    agent_dataset[-1].add_transitions(isample)
    
  group_samples['s1'] = np.concatenate(group_samples['s1'], axis=1)
  group_samples['s2'] = np.concatenate(group_samples['s2'], axis=1)
  group_samples['a1'] = np.concatenate(group_samples['a1'], axis=1)
  group_samples['a2'] = np.concatenate(group_samples['a2'], axis=1)
  group_samples['reward'] = np.sum(group_samples['reward'], axis=0)
  group_samples['discount'] = np.any(group_samples['discount'], axis=0)
  group_trans = Transition(s1=group_samples['s1'],
                           a1=group_samples['a1'],
                           s2=group_samples['s2'],
                           a2=group_samples['a2'],
                           reward=group_samples['reward'],
                           discount=group_samples['discount'])
  group_trans = numpy2tensor(group_trans)
  group_dataset.add_transitions(group_trans)
  
  n_train = min(n_train, group_dataset.size)
  logging.info('n_train %s.', n_train)
  if use_seed_for_data:
    rand = np.random.RandomState(seed)
  else:
    rand = np.random.RandomState(0)
  shuffled_indices = utils.shuffle_indices_with_steps(
      n=group_dataset.size, steps=shuffle_steps, rand=rand)
  train_indices = shuffled_indices[:n_train]
  group_data = group_dataset.create_view(train_indices)
  agents_data = []
  for i in range(num_agents):
    agents_data.append(agent_dataset[i].create_view(train_indices)) 
  group = group_set(batch_size, agents_data, group_data)
  # print(len(group.get_batch_samples()))
  # Prepare data.
  # logging.info('Loading data from %s ...', data_file)
  # data_size = utils.load_variable_from_ckpt(data_file, 'data._capacity')
  # with tf.device('/cpu:0'):
  #   full_data = dataset.Dataset(observation_spec, action_spec, data_size)
  # data_ckpt = tf.train.Checkpoint(data=full_data)
  # data_ckpt.restore(data_file)
  # # Split data.
  # n_train = min(n_train, full_data.size)
  # logging.info('n_train %s.', n_train)
  # if use_seed_for_data:
  #   rand = np.random.RandomState(seed)
  # else:
  #   rand = np.random.RandomState(0)
  # shuffled_indices = utils.shuffle_indices_with_steps(
  #     n=full_data.size, steps=shuffle_steps, rand=rand)
  # train_indices = shuffled_indices[:n_train]
  # train_data = full_data.create_view(train_indices)
  
  # Create agent.
  
  agents = []
  agent_ckpt_name = []
  initial_batch = group.get_batch_samples()
  for i in range(num_agents):
    agent_flags = utils.Flags(
      id=i,
      observation_spec=observation_spec,
      action_spec=action_spec,
      model_params=model_params,
      optimizers=optimizers,
      batch_size=batch_size,
      weight_decays=weight_decays,
      update_freq=update_freq,
      update_rate=update_rate,
      discount=discount,
      initial_batch=initial_batch[i])
  
    agent_args = agent_module.Config(agent_flags).agent_args
    agents.append(agent_module.Agent(**vars(agent_args)))
    agent_sub_dir = 'agent_' + str(i)
    agent_ckpt_name.append(os.path.join(log_dir, agent_sub_dir))

  # Restore agent from checkpoint if there exists one.
  # if tf.io.gfile.exists('{}.index'.format(agent_ckpt_name[0])):
  #   logging.info('Checkpoint found at %s.', agent_ckpt_name[0])
  #   for i, agent in enumerate(agents):
  #     agent.restore(agent_ckpt_name[i])

  # Train agent.
  # train_summary_dir = os.path.join(log_dir, 'train')
  # eval_summary_dir = os.path.join(log_dir, 'eval')
  # train_summary_writer = tf.compat.v2.summary.create_file_writer(
  #     train_summary_dir)
  # eval_summary_writers = collections.OrderedDict()
  # for policy_key in agent.test_policies.keys():
  #   eval_summary_writer = tf.compat.v2.summary.create_file_writer(
  #       os.path.join(eval_summary_dir, policy_key))
  #   eval_summary_writers[policy_key] = eval_summary_writer
  # eval_results = []

  time_st_total = time.time()
  time_st = time.time()
  step = agents[0].global_step
  timed_at_step = step
  while step < total_train_steps:
    agents_batch = group.get_batch_samples()
    extra_batch = group.get_batch_samples()
    for i in range(num_agents):
      # agents[i].train_step(agents_batch[i])   
      agents[i].train_step(agents_batch[i], extra_batch[i])   
    
    step = agents[0].global_step
    # if step % summary_freq == 0 or step == total_train_steps:
    #   agent.write_train_summary(train_summary_writer)
    if step % print_freq == 0 or step == total_train_steps:
      for agent in agents:
        agent.print_train_info()
    # if step % eval_freq == 0 or step == total_train_steps:
    #   time_ed = time.time()
    #   time_cost = time_ed - time_st
    #   logging.info(
    #       'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
    #   eval_result, eval_infos = train_eval_utils.eval_policies(
    #       tf_env, agent.test_policies, n_eval_episodes)
    #   eval_results.append([step] + eval_result)
    #   logging.info('Testing at step %d:', step)
    #   for policy_key, policy_info in eval_infos.items():
    #     logging.info(utils.get_summary_str(
    #         step=None, info=policy_info, prefix=policy_key+': '))
    #     utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
    #   time_st = time.time()
    #   timed_at_step = step
    if step % save_freq == 0:
      for i, agent in enumerate(agents):
        agent_name = os.path.join(agent_ckpt_name[i], 'agent'+str(step))
        agent.save(agent_name)
        logging.info('Agent %s saved at %s.', str(i), agent_name)

  for i, agent in enumerate(agents):
    agent_name = os.path.join(agent_ckpt_name[i], 'agent'+str(step))
    agent.save(agent_name)
  time_cost = time.time() - time_st_total
  logging.info('Training finished, time cost %.4gs.', time_cost)