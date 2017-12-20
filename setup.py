from setuptools import setup

setup(name='gym_duckietown',
      version='0.0.1',
      install_requires=[
        'gym',
        'visdom',
        'numpy',
        'scipy',
        'matplotlib'
      ]  # And any other dependencies foo needs
)