conda create --yes --name grounding python=3.9
conda activate grounding
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install .