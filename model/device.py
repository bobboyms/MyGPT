import torch


def get_device() -> torch.device:
    """
    Checks the available device for PyTorch operations and returns it.

    The function prioritizes devices in the following order: 
    MPS (Apple Silicon GPU), CUDA (NVIDIA GPU), and CPU.

    Returns:
    torch.device: The detected PyTorch device.
    """
    # Check for Apple Silicon GPU (MPS) availability
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # Check for NVIDIA GPU (CUDA) availability
    elif torch.cuda.is_available():
        return torch.device("cuda")

    # Default to CPU if neither MPS nor CUDA is available
    else:
        return torch.device("cpu")
