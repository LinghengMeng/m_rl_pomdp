from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The PL-POMDP repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='m_rl',
    py_modules=['m_rl'],
    version='0.1',
    install_requires=[
        'gymnasium[mujoco]>=0.29.0',
        # # 'cloudpickle==1.2.1',
        # # 'gym[atari,box2d,classic_control]~=0.15.3',
        # # 'ipython',
        'joblib>=1.3.1',
        # # 'matplotlib==3.1.1',
        # # 'mpi4py',
        # 'numpy',
        'pandas>=2.0.3',
        # # 'pytest',
        # # 'psutil',
        'scipy>=1.11.1',
        # # 'seaborn==0.8.1',
        # # 'tqdm',
        # 'google-cloud-storage',
        # 'torch',
        'SQLAlchemy',
        # 'pep517',
        # 'python-osc',
        # 'psycopg2'
    ],
    description="Code used for studying multistep bootstrapping and POMDP.",
    author="Lingheng Meng",
)
