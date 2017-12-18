from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import curses
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
import visdom

from gym_duckietown.envs.generate_map import generate_map


action_orientation = {
    0: prefab_sprites.MazeWalker._NORTH,
    1: prefab_sprites.MazeWalker._SOUTH,
    2: prefab_sprites.MazeWalker._WEST,
    3: prefab_sprites.MazeWalker._EAST,
}

GAME_ART = ['#############',
            '#     #     #',
            '#     #     #',
            '#     #     #',
            '#           #',
            '#     #     #',
            '#### ###### #',
            '#     #     #',
            '#     #     #',
            '#           #',
            '#     #     #',
            '# P   #     #',
            '#############']


def make_game():
    """Builds and returns a four-rooms game."""
    return ascii_art.ascii_art_to_game(
        generate_map(), what_lies_beneath=' ',
        sprites={'P': PlayerSprite})


class DuckietownGrid(gym.Env):

    def __init__(self, size=10):
        self.size = size
        map_art = generate_map(size)
        PlayerSprite._GAME_ART = map_art
        self.game = ascii_art.ascii_art_to_game(
            map_art, what_lies_beneath=' ',
            sprites={'P': PlayerSprite}
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(3)

    def _step(self, action):
        # Use the sprite position insteas of the whole board as an observation
        _, reward, _ = self.game.play(action)
        observation = self.game._sprites_and_drapes['P'].get_observation()
        return np.array(observation), reward, self.game.game_over, ""

    def _reset(self):
        # Find cleaner way to reset the end, for now just recreate it
        map_art = generate_map(self.size)
        PlayerSprite._GAME_ART = map_art
        self.game = ascii_art.ascii_art_to_game(
            map_art, what_lies_beneath=' ',
            sprites={'P': PlayerSprite}
        )
        _, reward, _ = self.game.its_showtime()
        observation = self.game._sprites_and_drapes['P'].get_observation()
        return observation

    def _render(self, mode="human", close=False):
        # raise NotImplementedError
        pass


class PlayerSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for our player.

    This `Sprite` ties actions to going in the four cardinal directions. If we
    reach a magical location (in this example, (4, 3)), the agent receives a
    reward of 1 and the epsiode terminates.
    """
    _GAME_ART = GAME_ART

    def __init__(self, corner, position, character, plot=False):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')
        self.orientation = self._NORTH
        self._create_memory()
        self.create_art()
        self.memory[position] = 1
        self.mask = np.array([-1, 0, 1])

        if plot:
            self.fig, self.ax = plt.subplots()
            self.vis = visdom.Visdom()
            self.win = self.vis.heatmap(self.memory)
        else:
            self.win = None

    def create_art(self):
        art = [list(line.replace('#', '1').replace(' ', '0').replace('P', '1')) for line in PlayerSprite._GAME_ART]
        art = [list(map(int, line)) for line in art]
        self.art = np.array(art)

    def _create_memory(self):
        memory = [list(line.replace('#', '1').replace(' ', '0').replace('P', '1')) for line in PlayerSprite._GAME_ART]
        memory = [list(map(int, line)) for line in memory]

        self.memory = np.array(memory)
        # n, m = selfmemory.shape

    def _update_memory(self, the_plot=None):
        window = np.multiply.outer(self.orientation[::-1], self.mask).T + self.position + self.orientation
        u, v = window.T
        before = self.memory[u, v].sum()
        self.memory[u, v] = 1
        after = self.memory[u, v].sum()
        #the_plot.add_reward((after - before)/(3*5))
        the_plot.add_reward(after-before)

    def get_observation(self):
        window = np.multiply.outer(self.orientation[::-1], self.mask).T + self.position + self.orientation
        u, v = window.T
        return self.art[u, v]

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things   # Unused.

        # Apply motion commands.
        if actions is not None and actions < 4 and self.orientation != action_orientation[actions]:
            self.orientation = action_orientation[actions]      
        elif actions == 0:
            out = self._north(board, the_plot)
        elif actions == 1:
            out = self._south(board, the_plot)
        elif actions == 2:
            out = self._west(board, the_plot)
        elif actions == 3:
            out = self._east(board, the_plot)

            if out is not None:
                the_plot.add_reward(-0.96)

        # Penalty for each step taken
        the_plot.add_reward(-0.04)

        self._update_memory(the_plot)

        # See if we've explored the map
        if np.all(self.memory == 1):
            the_plot.add_reward(10.0)
            the_plot.terminate_episode()

        if self.win is not None:
            self.vis.heatmap(self.memory, win=self.win)


def main(argv=()):
    del argv  # Unused.

    # Build a four-rooms game.
    game = make_game()

    # Make a CursesUi to play it with.
    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                -1: 5},
        delay=200)

    # Let the game begin!
    ui.play(game)

if __name__ == '__main__':
    main(sys.argv)
