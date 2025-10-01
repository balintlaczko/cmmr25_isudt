import torch
from .tensor import scale


def square_over_bg_falloff(
    x: int,
    y: int,
    img_size: int = 64,
    square_size: int = 2,
    falloff_mult: int = 0.1,
    color: list = None,
) -> torch.Tensor:
    """
    Create an image of a square over a black background with a falloff.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 64.
        square_size (int, optional): The size of each side of the square. Defaults to 2.
        falloff_mult (int, optional): The falloff multiplier. Defaults to 0.1.
        color (list, optional): A color vector to multiply the image with. If None, single-channel images are generated. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1
    # create falloff
    falloff = torch.zeros((img_size, img_size))
    _x, _y = x + square_size / 2, y + square_size / 2
    i, j = torch.meshgrid(torch.arange(img_size), torch.arange(img_size))
    v_to_square = torch.stack(
        [i, j]) - torch.tensor([_y, _x], dtype=torch.float32).view(2, 1, 1)
    v_length = torch.norm(v_to_square, dim=0)
    falloff = 1 - torch.clip(scale(v_length, 0, img_size,
                             0, img_size, exp=falloff_mult) / img_size, 0, 1)
    img = torch.clip(img + falloff, 0, 1)
    if color != None:
        img = img.unsqueeze(-1) * torch.tensor(color)

    return img
