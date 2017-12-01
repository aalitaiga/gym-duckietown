from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import numpy as np
import visdom
import matplotlib.pyplot as plt
# plt.ion()
import seaborn as sns; sns.set()

from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites


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

# Rewrite the engine function and replace engine.play by engine.step


def make_game():
    """Builds and returns a four-rooms game."""
    return ascii_art.ascii_art_to_game(
        GAME_ART, what_lies_beneath=' ',
        sprites={'P': PlayerSprite})


class DuckietownGrid(gym.Env):

    def __init__(self):
        self.game = ascii_art.ascii_art_to_game(
            GAME_ART, what_lies_beneath=' ',
            sprites={'P': PlayerSprite}
        )
        self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        # self.observation_space 

    def _step(self, action):
        # Use the sprite position insteas of the whole board as an observation
        _, reward, _ = self.game.play(action)
        sprite_position = self.game._sprites_and_drapes['P'].virtual_position
        return np.array(sprite_position), reward, self.game.game_over, ""

    def _reset(self):
        # Find clearner way to reset the end, for now just recreate it
        self.game = ascii_art.ascii_art_to_game(
            GAME_ART, what_lies_beneath=' ',
            sprites={'P': PlayerSprite}
        )
        observation, reward, _ = self.game.its_showtime()
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

    def __init__(self, corner, position, character, plot=True):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')
        self.orientation = self._NORTH
        self._create_memory()
        self.memory[position] = 1
        self.mask = np.array([-1, 0, 1])
        self.fig, self.ax = plt.subplots()

        if plot:
            self.vis = visdom.Visdom()
            self.win = self.vis.heatmap(self.memory)
        else:
            self.win = None

    def _create_memory(self):
        memory = [list(line.replace('#', '1').replace(' ', '0').replace('P', '1')) for line in GAME_ART]
        memory = [map(int, line) for line in memory]
        self.memory = np.array(memory)
        # n, m = selfmemory.shape

    def _update_memory(self):
        window = np.multiply.outer(self.orientation[::-1], self.mask).T + self.position + self.orientation
        u, v = window.T
        self.memory[u, v] = 1

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things   # Unused.

        # Apply motion commands.
        if actions in [0, 1, 2, 3]:    # walk upward?
            self.orientation = action_orientation[actions]
        elif actions == 4:
            # TODO: add penalty for illegal moves
            # Return None if fine, else return blocking object
            if self.orientation == self._NORTH:
                self._north(board, the_plot)
            elif self.orientation == self._SOUTH:
                self._south(board, the_plot)
            elif self.orientation == self._EAST:  # walk downward?
                self._east(board, the_plot)
            elif self.orientation == self._WEST:  # walk leftward?
                self._west(board, the_plot)

        self._update_memory()

        # See if we've explored the map
        if np.all(self.memory == 1):
            the_plot.add_reward(1.0)
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
                curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3, 'f': 4,
                -1: 5},
        delay=200)

    # Let the game begin!
    ui.play(game)

if __name__ == '__main__':
    main(sys.argv)
