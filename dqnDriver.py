from gym.version import VERSION
print(VERSION) # make sure the new version of gym is loaded
import gym
import numpy as np
import tensorflow as tf
import pandas as pd


class Agent:
  def __init__(self, obs_shape, act_size):
    self.obs_shape = obs_shape
    self.act_size = act_size
  def network(self, train=True, cust_layer_nums = [64,64], act_func = tf.keras.layers.LeakyReLU()):
    inputs = tf.keras.Input(shape=(self.obs_shape,), name="input")
    x=None
    for i,l in enumerate(cust_layer_nums):
      if i==0:
        x = tf.keras.layers.Dense(cust_layer_nums[i], activation=act_func, name=f"dense_{i+1}")(inputs)
      else:
        x = tf.keras.layers.Dense(cust_layer_nums[i], activation=act_func, name=f"dense_{i+1}")(x)
    outputs = tf.keras.layers.Dense(self.act_size, name="output")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="nn", trainable=train)
    return model
class Util:
  def __init__(self, init_l_rate=0.01):
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(init_l_rate, decay_steps=50, decay_rate=0.9))
    self.history = []
  def record_history(self, current_state, action, reward, next_state):
    self.history.append([current_state, action, reward, next_state])
  def td_loss(self, nn, discount=0.99):
    loss = []
    for current_state, action, reward, next_state in self.history:
      binary_action = [0.0] * nn.output.shape[1]
      binary_action[action] = 1.0
      binary_action = tf.constant([binary_action])
      q_current = nn(tf.convert_to_tensor([current_state]))
      max_q_next = tf.math.reduce_max(nn(tf.convert_to_tensor([next_state])))
      loss.append(tf.math.square((reward + discount * max_q_next - q_current) * binary_action))
    return tf.math.reduce_mean(loss, axis=0)
  
  def update_model(self, nn):
    with tf.GradientTape() as tape:
      loss = self.td_loss(nn)
    grads = tape.gradient(loss, nn.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, nn.trainable_variables))
    self.history = []

def run_experiment(cust_layer_nums = [128,64], start_epsilon = 0.3, act_func = tf.keras.layers.LeakyReLU(), act_func_name="leaky_relu", num_games = 2000, l_rate=0.01):
  env = gym.make("CartPole-v1")
  agent = Agent(4, 2).network(cust_layer_nums=cust_layer_nums, act_func=act_func)
  utility = Util(init_l_rate=l_rate)
  # train
  epsilon = start_epsilon
  i, early_stop = 0, 0
  n_game = num_games

  steps_per_game = []

  while i < n_game:
    current_state = env.reset()
    step = 0
    while True:
      if np.random.uniform() < epsilon:
        action = env.action_space.sample()
      else:
        action = tf.math.argmax(tf.reshape(agent(tf.convert_to_tensor([current_state])), [-1])).numpy()
      next_state, reward, done, info = env.step(action)
      step += 1
      utility.record_history(current_state, action, reward, next_state)
      current_state = next_state
      if len(utility.history) == 50:
        utility.update_model(agent)
      epsilon = max(epsilon * 0.99, 0.05)
      if done:
        #print(i, step)
        if i % int(num_games/5) == 0:
          print(f"{i/(num_games/100)}% done")
        steps_per_game.append(step)
        i += 1
        if step >= 500:
          early_stop += 1
        else:
          early_stop = 0
        if early_stop >= 10:
          i = n_game
        break
  # test
  test_step=0
  test_util=0
  for i in range(10):
    env.close()
    env = gym.make("CartPole-v1")
    state = env.reset()
    step = 0
    while True:
      action = tf.math.argmax(tf.reshape(agent(tf.convert_to_tensor([state])), [-1])).numpy()
      state, reward, done, info = env.step(action)
      test_util+=reward
      step += 1
      if done:
        test_step+=step
        #print(i, step)
        
        break
  env.close()

  return steps_per_game, agent, test_step, test_util
# save agent
# agent.save("cartpole_dql")

cust_layer_nums = [
  [32],
  [32,4],
  [16,16],
  [8,8,8,8],
  [32,32],
  [64,64],
  [128,32,16],
  [256,64,16]
]
start_epsilons = [
  0.3,
  1
]
l_rates = [
  0.005,
  0.01,
  0.05,
  0.1,
  0.5
]
act_funcs = [
  tf.keras.layers.LeakyReLU(),  
  tf.keras.activations.sigmoid,
  tf.keras.activations.tanh,
  tf.keras.activations.elu
]
act_names = [
  "LeakyReLU",  
  "sigmoid",
  "tanh",
  "elu"
]
game_lens = [
  250,500,1000,2000,4000,6000
]

b_arch = [24,24]
b_eps = 0.3
b_lr = 0.01
b_act = tf.keras.activations.sigmoid
b_act_name = "relu"
b_games = 0

b_steps = 0
b_step_ai = None
b_util = 0
b_util_ai = None

results = pd.DataFrame(columns=["test_steps","test_util","architecture","Epsilon","activation","games_played","l_rate","steps"])
results = pd.read_csv("results_temp.csv")


steps_per_game, agent, test_step, test_util = run_experiment(cust_layer_nums = [24,24], start_epsilon = 0.3, act_func = tf.keras.layers.LeakyReLU(), act_func_name="leaky_relu", num_games = 2000, l_rate=0.01)
tempdict = {}
tempdict["test_steps"] = test_step
tempdict["test_util"] = test_util
tempdict["architecture"] = [24,24]
tempdict["Epsilon"] = 0.3
tempdict["activation"] = "leaky_relu"
tempdict["games_played"] = 2000
tempdict["l_rate"] = 0.01
tempdict["steps"] = steps_per_game
b_steps = test_step
b_step_ai = tempdict
b_util = test_util
b_util_ai = tempdict
results = results.append(tempdict, ignore_index=True)
#print(tempdict)
print(f"finished a run \n{results.loc[:, results.columns != 'steps']}")



for l in cust_layer_nums:
  steps_per_game, agent, test_step, test_util = run_experiment(cust_layer_nums = l, start_epsilon = 0.3, act_func = tf.keras.layers.LeakyReLU(), act_func_name="leaky_relu", num_games = 2000, l_rate=0.01)
  tempdict = {}
  tempdict["test_steps"] = test_step
  tempdict["test_util"] = test_util
  tempdict["architecture"] = l
  tempdict["Epsilon"] = 0.3
  tempdict["activation"] = "leaky_relu"
  tempdict["games_played"] = 2000
  tempdict["l_rate"] = 0.01
  tempdict["steps"] = steps_per_game
  results = results.append(tempdict, ignore_index=True)
  #print(tempdict)
  print(f"finished a run \n{results.loc[:, results.columns != 'steps']}")

  if test_step > b_steps:
    b_steps = test_step
    b_step_ai = tempdict
    b_arch = l
  if test_util > b_util:
    b_util = test_util
    b_util_ai = tempdict

  results.to_csv("results_temp.csv")  

print("Done with architectures, Printing best AI")
print(f"Steps: {results.loc[:, results.columns != 'steps']}")
print(f"Util: {results.loc[:, results.columns != 'steps']}")

for se in start_epsilons:
  steps_per_game, agent, test_step, test_util = run_experiment(cust_layer_nums = b_arch, start_epsilon = se, act_func = tf.keras.layers.LeakyReLU(), act_func_name="leaky_relu", num_games = 2000, l_rate=0.01)
  tempdict = {}
  tempdict["test_steps"] = test_step
  tempdict["test_util"] = test_util
  tempdict["architecture"] = b_arch
  tempdict["Epsilon"] = se
  tempdict["activation"] = "leaky_relu"
  tempdict["games_played"] = 2000
  tempdict["l_rate"] = 0.01
  tempdict["steps"] = steps_per_game
  results = results.append(tempdict, ignore_index=True)

  if test_step > b_steps:
    b_steps = test_step
    b_step_ai = tempdict
    b_eps = se
  if test_util > b_util:
    b_util = test_util
    b_util_ai = tempdict

  results.to_csv("results_temp.csv")  


for lr in l_rates:
  steps_per_game, agent, test_step, test_util = run_experiment(cust_layer_nums = b_arch, start_epsilon = b_eps, act_func = tf.keras.layers.LeakyReLU(), act_func_name="leaky_relu", num_games = 2000, l_rate=lr)
  tempdict = {}
  tempdict["test_steps"] = test_step
  tempdict["test_util"] = test_util
  tempdict["architecture"] = b_arch
  tempdict["Epsilon"] = b_eps
  tempdict["activation"] = "leaky_relu"
  tempdict["games_played"] = 2000
  tempdict["l_rate"] = lr
  tempdict["steps"] = steps_per_game
  results = results.append(tempdict, ignore_index=True)

  if test_step > b_steps:
    b_steps = test_step
    b_step_ai = tempdict
    b_lr = lr
  if test_util > b_util:
    b_util = test_util
    b_util_ai = tempdict

  results.to_csv("results_temp.csv") 


for act in range(len(act_funcs)):
  steps_per_game, agent, test_step, test_util = run_experiment(cust_layer_nums = b_arch, start_epsilon = b_eps, act_func = act_funcs[act], act_func_name="leaky_relu", num_games = 2000, l_rate=b_lr)
  tempdict = {}
  tempdict["test_steps"] = test_step
  tempdict["test_util"] = test_util
  tempdict["architecture"] = b_arch
  tempdict["Epsilon"] = b_eps
  tempdict["activation"] = act_names[act]
  tempdict["games_played"] = 2000
  tempdict["l_rate"] = b_lr
  tempdict["steps"] = steps_per_game
  results = results.append(tempdict, ignore_index=True)

  if test_step > b_steps:
    b_steps = test_step
    b_step_ai = tempdict
    b_act_name = act_names[act]
    b_act = act_funcs[act]
  if test_util > b_util:
    b_util = test_util
    b_util_ai = tempdict

  results.to_csv("results_temp.csv") 

b_arch = [24,24]
b_eps = 0.3
b_lr = 0.01
b_act = tf.keras.activations.sigmoid
b_act_name = "leaky_relu"
b_games = 0

b_steps = 0
b_step_ai = None
b_util = 0
b_util_ai = None
for g in game_lens:
  steps_per_game, agent, test_step, test_util = run_experiment(cust_layer_nums = b_arch, start_epsilon = b_eps, act_func = b_act, act_func_name=b_act_name, num_games = g, l_rate=b_lr)
  tempdict = {}
  tempdict["test_steps"] = test_step
  tempdict["test_util"] = test_util
  tempdict["architecture"] = b_arch
  tempdict["Epsilon"] = b_eps
  tempdict["activation"] = b_act_name
  tempdict["games_played"] = g
  tempdict["l_rate"] = b_lr
  tempdict["steps"] = steps_per_game
  results = results.append(tempdict, ignore_index=True)

  if test_step > b_steps:
    b_steps = test_step
    b_step_ai = tempdict
    b_games = g
  if test_util > b_util:
    b_util = test_util
    b_util_ai = tempdict

  results.to_csv("results_temp.csv") 
