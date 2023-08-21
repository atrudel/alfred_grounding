import os.path
from config import REPO_ROOT


def test_data_full_is_available():
    assert os.path.isfile(REPO_ROOT / 'alfred/data/full_2.1.0/train/pick_heat_then_place_in_recep-PotatoSliced-None-Fridge-20/trial_T20190909_123440_684188/raw_images/000000000.jpg')
