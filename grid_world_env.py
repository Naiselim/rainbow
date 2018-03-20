from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from collections import deque
from ai_safety_gridworlds.environments import island_navigation
from ai_safety_gridworlds.environments.shared.safety_game import Actions

import copy
# glb_counter = 0


#def preprocess_frame(observ, output_size):
#    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
#    output = cv2.resize(gray, (output_size, output_size))
#    output = output.astype(np.float32, copy=False)
#    return output


class Environment(object):
    def __init__(self,
                 rom_file,
                 frame_skip,
                 num_frames,
                 frame_size,
                 no_op_start,
                 rand_seed,
                 dead_as_eoe, #?
                 env_name = 'island_navigation'):
#        self.ale = self._init_ale(rand_seed, rom_file)
        # normally (160, 210)

        if env_name == 'island_navigation':
            self.env = island_navigation.IslandNavigationEnvironment()

        timestep = self.env.reset()
        self.map = np.array(timestep.observation['board'])
        self.action_example = [1]
    # Get all allowed actions.
        self.actions_dict = {0: Actions.LEFT.value, 1: Actions.RIGHT.value, 2: Actions.UP.value, 3: Actions.DOWN.value}
        self.last_timestep = timestep

    #    self.actions = self.ale.getMinimalActionSet()

        self.frame_skip = frame_skip
        self.num_frames = num_frames
#        print (timestep.observation['board'])
        self.frame_size = np.shape(timestep.observation['board'])
#        print (self.frame_size)
#        exit(0)
        self.no_op_start = no_op_start
        self.dead_as_eoe = dead_as_eoe

        self.clipped_reward = 0
        self.total_reward = 0
    #    screen_width, screen_height = self.ale.getScreenDims()
    #    self.prev_screen = np.zeros(
    #        (screen_height, screen_width, 3), dtype=np.float32)
        self.frame_queue = deque(maxlen=num_frames)
        self.end = True

#    @staticmethod
#    def _init_ale(rand_seed, rom_file):
#        assert os.path.exists(rom_file), '%s does not exists.'
#        ale = ALEInterface()
#        ale.setInt('random_seed', rand_seed)
#        ale.setBool('showinfo', False)
#        ale.setInt('frame_skip', 1)
#        ale.setFloat('repeat_action_probability', 0.0)
#        ale.setBool('color_averaging', False)
#        ale.loadROM(rom_file)
#        return ale
    def print_map(self):
	print (self.last_timestep.observation['board'])

    @property
    def num_actions(self):
        return len(self.actions_dict)

    def _get_current_frame(self):
        # global glb_counter
        # screen = self.ale.getScreenRGB()
        # max_screen = np.maximum(self.prev_screen, screen)
        # frame = preprocess_frame(max_screen, self.frame_size)
        # frame /= 255.0
        # cv2.imwrite('test_env/%d.png' % glb_counter, cv2.resize(frame, (800, 800)))
        # glb_counter += 1
        # print 'glb_counter', glb_counter
        return self.last_timestep.observation['board']

    def reset(self):
        for _ in range(self.num_frames - 1):
            self.frame_queue.append(
                np.zeros(self.frame_size, dtype=np.float32))

    #    self.ale.reset_game()
    #    self.env.reset()
        self.clipped_reward = 0
        self.total_reward = 0
    #    self.prev_screen = np.zeros(self.prev_screen.shape, dtype=np.float32)

    #    n = np.random.randint(0, self.no_op_start)
    #    for i in range(n):
    #        if i == n - 1:
    #            self.prev_screen = self.ale.getScreenRGB()
    #        self.ale.act(0)

    #    timestep =
        self.last_timestep = self.env.reset()
        state = self.last_timestep.observation['board']

        self.frame_queue.append(self._get_current_frame())
    #    assert not self.ale.game_over()
        self.end = False

        return np.array(self.frame_queue)

    def step(self, action_idx):
        """Perform action and return frame sequence and reward.
        Return:
        state: [frames] of length num_frames, 0 if fewer is available
        reward: float
        """
        assert not self.end
        reward = 0
        clipped_reward = 0
#        old_lives = self.ale.lives()

        for _ in range(self.frame_skip):
    #        self.prev_screen = self.ale.getScreenRGB()
            timestep = self.env.step(self.actions_dict[action_idx])

            state = timestep.observation['board']
            r = timestep.reward
            done = self.env._game_over
            # r = self.ale.act(self.actions[action_idx])
            reward += r
            clipped_reward += np.sign(r)
    #        dead = (self.ale.lives() < old_lives)
            self.timestep = timestep
            if done:
                self.end = True
                break

        self.frame_queue.append(self._get_current_frame())
        self.total_reward += reward
        self.clipped_reward += clipped_reward
        return np.array(self.frame_queue), clipped_reward

    def print_map(self):
        print (self.timestep.observation['board'])

if __name__ == '__main__':

    env = Environment('roms/breakout.bin', 1, 1, 0, 0, 0, False)
    print ('starting with game over?', env.end)

    state = env.reset()
    i = 0
    while not env.end:
        #print (i)
        action = np.random.randint(0, env.num_actions)
        state, reward = env.step(action)
	print (i, reward, action)
        env.print_map()
        #if i % 100 == 0:

        #    for idx, f in enumerate(state):
        #        filename = 'test_env/f%d-%d.png' % (i, idx)
        #        cv2.imwrite(filename, cv2.resize(f, (800, 800)))
        i += 1
    print ('total_reward:', env.total_reward)
    print ('clipped_reward', env.clipped_reward)
    print ('total steps:', i)
