
def shuffle_down(x, factor):
    # format: (B, C, H, W)
    b, c, h, w = x.shape

    assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of " + str(factor) + "!"

    n = x.reshape(b, c, int(h/factor), factor, int(w/factor), factor)
    n = n.permute(0, 3, 5, 1, 2, 4)
    n = n.reshape(b, c*factor**2, int(h/factor), int(w/factor))

    return n


def shuffle_up(x, factor):
    # format: (B, C, H, W)
    b, c, h, w = x.shape

    assert c % factor**2 == 0, "C must be a multiple of " + str(factor**2) + "!"

    n = x.reshape(b, factor, factor, int(c/(factor**2)), h, w)
    n = n.permute(0, 3, 4, 1, 5, 2)
    n = n.reshape(b, int(c/(factor**2)), factor*h, factor*w)

    return n
