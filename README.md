# tf2_TileLayer

Split a 4D-Tensor into non-overlapping Tiles

Tile2D:

Reshapes a Tensor with shape [Batch_size, h, w, channels] into a Tensor with shape [Batch_size, n*n, h/n, w/n, channels]

Untile2D:

Reshapes a Tensor with shape [Batch_size, n* n, h, w, channels] into a Tensor with shape [Batch_size, h* n, w* n, channels]

Arguments:

n: Number of vertial and horizontal tiles. Input.shape[1] and Input.shape[2] (height & width) must be divisble by n to work proberly. 
