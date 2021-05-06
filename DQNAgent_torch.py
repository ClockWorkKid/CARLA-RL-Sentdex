import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from threading import Thread
from tqdm import tqdm

try:
    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Custom"

MEMORY_FRACTION = 0.8
MIN_REWARD = -100

EPISODES = 10000

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        spectator = self.world.get_spectator()
        world_snapshot = self.world.wait_for_tick()
        spectator.set_transform(self.vehicle.get_transform())
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))


        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=.5, steer=-0.5))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


class NeuralNet(nn.Module):
    def __init__(self, input_size=(3, 480, 640), output_size=3):
        super(NeuralNet, self).__init__()
        shape = np.array([input_size[1], input_size[2]]).astype(np.int32)
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=256, kernel_size=3)
        shape = shape - 2
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        shape = shape//2
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        shape = shape - 2
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        shape = shape//2
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        shape = shape - 2
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        shape = shape//2
        self.drop3 = nn.Dropout(0.2)

        self.dens1 = nn.Linear(in_features=256*shape[0]*shape[1], out_features=128)
        self.dens2 = nn.Linear(128, 64)
        self.dens3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)   # (n, 3, 480, 640)    ->  (n, 256, 478, 638)
        x = self.relu1(x)
        x = self.pool1(x)   # (n, 256, 478, 638)  ->  (n, 256, 239, 319)
        x = self.drop1(x)

        x = self.conv2(x)   # (n, 256, 239, 319)  ->  (n, 256, 237, 317)
        x = self.relu2(x)
        x = self.pool2(x)   # (n, 256, 237, 317)  ->  (n, 256, 118, 158)
        x = self.drop2(x)

        x = self.conv3(x)   # (n, 256, 118, 158)  ->  (n, 256, 116, 156)
        x = self.relu3(x)
        x = self.pool3(x)   # (n, 256, 116, 156)  ->  (n, 256, 58, 78)
        x = self.drop3(x)

        # x = x.view(x.size(0), -1)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.dens1(x)
        x = self.dens2(x)
        x = self.dens3(x)

        return x


class DQNAgent:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("Device" + self.device)

        self.model = NeuralNet(input_size=(3, IM_HEIGHT, IM_WIDTH), output_size=3)
        self.target_model = NeuralNet(input_size=(3, IM_HEIGHT, IM_WIDTH), output_size=3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        self.step = 0

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_states = np.swapaxes(current_states, 1, 3)
        current_states = np.swapaxes(current_states, 2, 3)
        current_states = torch.Tensor(current_states.astype(np.float32))
        current_qs_list = self.model(current_states.to(self.device))
        current_qs_list = current_qs_list.cpu().detach().numpy()

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        new_current_states = np.swapaxes(new_current_states, 1, 3)
        new_current_states = np.swapaxes(new_current_states, 2, 3)
        new_current_states = torch.Tensor(new_current_states.astype(np.float32))
        future_qs_list = self.target_model(new_current_states.to(self.device))
        future_qs_list = future_qs_list.cpu().detach().numpy()

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(np.rollaxis(current_state, 2, 0)/255)
            y.append(current_qs)

        X = torch.Tensor(np.array(X).astype(np.float32))
        y = torch.Tensor(np.array(y))

        log_this_step = False
        if self.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.step

        """ Train and Fit Model """
        y_pred = self.model(X.to(self.device)).cpu()
        loss = self.loss_function(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        state = np.rollaxis(state, 2, 0)  # shape of (H, W, C) to (C, H, W)
        state = np.array(state).reshape(-1, *state.shape)  # (C, H, W) to (1, C, H, W)
        state = state.astype(np.float32)
        state = torch.Tensor(state / 255)
        prediction = self.model(state.to(self.device)).cpu().detach().numpy()
        return prediction[0]

    def train_in_loop(self):
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    print("Beginning Simulation")
    FPS = 60
    # For stats

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    torch.random.manual_seed(1)


    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    ep_rewards = []
    avg_rewards = []
    max_rewards = []
    min_rewards = []

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
        env.collision_hist = []

        # Update tensorboard step every episode
        agent.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, 3)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            avg_rewards.append(average_reward)
            max_rewards.append(max_reward)
            min_rewards.append(min_reward)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                torch.save(agent.model.state_dict(),
                           f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pt')

    # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()

    print("Finished training model")
    plt.figure()
    plt.subplot(211)
    plt.plot(max_rewards, label="max")
    plt.plot(avg_rewards, label="avg")
    plt.plot(min_rewards, label="min")
    plt.legend(loc=4)
    plt.subplot(212)
    plt.plot(ep_rewards)
    plt.show()
