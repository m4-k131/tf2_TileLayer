# -*- coding: utf-8 -*-
"""
@author: Malte Kleine
"""
from tensorflow import reshape, transpose
from tensorflow.keras import layers

class Tile2D(layers.Layer):
    def __init__(self, n):
        super(Tile2D, self).__init__()
        self.n = n
        
    def build(self, input_shape):
        self.tw = input_shape[1]//self.n
        self.th = input_shape[2]//self.n
        self.channels = input_shape[3]
        self.nsqrd=self.n**2
 
    def call(self, images):
        tiles = reshape(images, (-1 ,self.n, self.tw, self.n, self.th, self.channels))
        tiles = transpose(tiles, [0, 1, 3, 2, 4, 5])
        tiles = reshape(tiles, (-1, self.nsqrd, self.tw, self.th, self.channels))
        return tiles

class Untile2D(layers.Layer):
    def __init__(self, n):
        super(Untile2D, self).__init__()
        self.n = n
        
    def build(self, input_shape):
        self.tw = input_shape[2]
        self.th = input_shape[3]
        self.iw = self.tw*self.n
        self.ih = self.th*self.n
        self.channels = input_shape[4]
                           
    def call(self, tiles):
        images = reshape(tiles, (-1, self.n, self.n, self.tw, self.th, self.channels))
        images = transpose(images, [0, 1, 3, 2, 4, 5])
        images = reshape(images, (-1, self.iw, self.ih, self.channels))
        return images
