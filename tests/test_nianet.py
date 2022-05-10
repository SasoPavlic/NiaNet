from nianet import __version__
import torch


def test_version():
    assert __version__ == '1.0.0'

def test_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

