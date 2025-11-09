import torch
import pytest


@pytest.fixture(scope="session", autouse=True)
def _seed():
    torch.manual_seed(1234)


@pytest.fixture(scope="session")
def device():
    return "cpu"
