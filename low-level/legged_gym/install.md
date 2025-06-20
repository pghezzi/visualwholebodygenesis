# Create a new python virtual env with python>=3.10
# Install PyTorch
# Install Genesis following the instructions in the Genesis repo

# Install rsl_rl.
git clone git@github.com:leggedrobotics/rsl_rl.git
cd rsl_rl && git checkout v1.0.2 && pip install -e . --use-pep517

# Install tensorboard.
pip install tensorboard