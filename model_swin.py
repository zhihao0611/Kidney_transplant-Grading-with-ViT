# File: model.py
# Author: Zhihao Huang
# Notes: kidney transplantation

# %% 0. Set up
# 0.a Package imports
from mil_new.metrics import Balanced, F1, Mcc, Sensitivity, Specificity
from mil_new.models import convolutional_model, attention_flat, attention_flat_tune
from mil_new.io.reader import read_record, peek
from mil_new.io.transforms import parallel_dataset
from mil_new.io.utils import inference, study
from mil_new.io.writer import write_record
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import ray
from keras import layers
from tensorflow import keras
from keras import backend as K

# feature extraction parameters
t=224 # tile size (pixels)
overlap=112 # tile overlap (pixels)
chunk=1792 # chunk size (pixels)
magnification=20 # magnification
tile_batch=128 # the number of tiles to batch
tile_prefetch=2 # the number of batches to prefetch

# path parameters
nwu_svspath = '/data/northwestern/wsi/' # path for the northwestern whole-slide images
nwu_csvfile = '/data/northwestern/CTOT08_clinical_BiopsyImageKeys_4.27.22.csv'
nwu_features = '/data/northwestern/features/'
emory_csvfile = '/data/emory/PAS_TransplantSlideManifest.csv'
emory_features = '/data/emory/features/'

# training parameters
tau = 2.0 # threshold for binarizing banff scores
train_size = 0.8 # 20% validation set
test_split_state = 1
optimization = {'epochs': 10,
                'lr': 1e-5,
                'num_updates': 5,
                'decay': 0.99}

# other parameters
emory_mapping = {'name': 'name', 'meta.g_score': 'g', 'meta.ptc_score': 'ptc',
                  'meta.v_score': 'v', 'meta.cg_score': 'cg', 'meta.mm_score': 'mm', 
                  'meta.ci_score': 'ci', 'meta.cv_score': 'cv', 
                  'meta.i_score': 'i', 'meta.t_score': 't', 'meta.ah_score': 'ah'}
nwu_mapping = {'SVS_FileName': 'name', 'G': 'g', 'PTC': 'ptc',
                  'V': 'v', 'TG': 'tg', 'CG': 'cg', 'MM': 'mm', 'CI': 'ci',
                  'CT': 'ct', 'CV': 'cv', 'I': 'i', 'T': 't', 'AH': 'ah'}
emory_scores = [emory_mapping[key] for key in emory_mapping.keys()]
nwu_scores = [nwu_mapping[key] for key in nwu_mapping.keys()]
overlap_scores = list(set(emory_scores).intersection(set(nwu_scores) - {'name'}))
# model = 'EfficientNetV2S_224_112_20X'
# D = 1280 # EfficientNetV2 output shape (both 'S' and 'L' models)
model = 'EfficientNetV2S_224_112_20X_fp16'
D = 1280 # convnext output shape

# get a list of emory files for the given model
emory_files = [f'{emory_features}/{model}/{file}' for file in os.listdir(f'{emory_features}/{model}/') if os.path.splitext(file)[1] == '.tfr']
nwu_files = [f'{nwu_features}/{model}/{file}' for file in os.listdir(f'{nwu_features}/{model}/') if os.path.splitext(file)[1] == '.tfr']

# read the tables
emory_table = pd.read_csv(emory_csvfile)
nwu_table = pd.read_csv(nwu_csvfile)

# eliminate entries where .tfr files are not available
f = lambda file: os.path.split(file)[1].split('.svs')[0]+'.svs'
emory_trimmed = [f(file) for file in emory_files]
emory_table = emory_table.loc[emory_table['name'].isin(emory_trimmed),:]
nwu_trimmed = [f(file) for file in nwu_files]
nwu_table = nwu_table.loc[nwu_table['SVS_FileName'].isin(nwu_trimmed),:]

# for each score, generate the list of files where the score is available, and generate stratified training, testing
datasets = {}
for score in overlap_scores:
    
    # find key in dictionaries
    f = lambda d, v: next(key for key, value in d.items() if value == v)
    nwu_key = f(nwu_mapping, score)
    emory_key = f(emory_mapping, score)
    
    # filter tables to where score is available
    emory_select = emory_table.loc[~emory_table[emory_key].isnull()]
    nwu_select = nwu_table.loc[~nwu_table[nwu_key].isnull()]
    
    # get scores and .svs filenames
    emory_select_scores = np.array(emory_select[emory_key])
    emory_select_files = list(emory_select['name'])
    nwu_select_files = list(nwu_select['SVS_FileName'])
    
    # calculate class weight
    weight = class_weight.compute_class_weight('balanced',
                                  classes=np.unique(np.unique(emory_select_scores>=tau)),
                                  y=emory_select_scores>=tau)
    weight = {k:v for k, v in zip([0, 1], weight)}
    print('class weight: ', weight)

    # stratify emory data
    train, validation = train_test_split(emory_select_files, 
                                         train_size=train_size, 
                                         stratify=emory_select_scores>=tau)
    
    # build .tfr filenames from .svs filenames
    f = lambda file, model: f'{file}.{model}.tfr'
    train = [f(t, model) for t in train]
    validation = [f(v, model) for v in validation]
    test = [f(t, model) for t in nwu_select_files]
    
    # capture datasets
    datasets[score] = {'train': [f'{emory_features}{model}/{t}' for t in train],
                       'validation': [f'{emory_features}{model}/{v}' for v in validation],
                       'test': [f'{nwu_features}{model}/{t}' for t in test],
                       'class_weight': weight}

# get lists of .tfr variables for de-serialization
import json
serialized = list(tf.data.TFRecordDataset(datasets[overlap_scores[0]]['test'][0]))[0]
nwu_variables = peek(serialized)
features, _, _, _ = read_record(serialized, nwu_variables, structured=True)
print(f"test_features (structured): \n\t{features.shape}, {features.dtype}")

serialized = list(tf.data.TFRecordDataset(datasets[overlap_scores[0]]['train'][0]))[0]
emory_variables = peek(serialized)
features, _, _, _ = read_record(serialized, emory_variables, structured=True)
print(f"train_features (structured): \n\t{features.shape}, {features.dtype}")

# max_width = 0
# max_height = 0
# for i in range(428):
#   serialized = list(tf.data.TFRecordDataset(datasets[overlap_scores[0]]['train'][i]))[0]
#   emory_variables = peek(serialized)
#   features, _, _, _ = read_record(serialized, emory_variables, structured=True)
#   if features.shape[0] > max_width:
#     max_width = features.shape[0]
#   if features.shape[1] > max_height:
#     max_height = features.shape[1]
# print(f"\ntrain_max_width: \n\t{max_width}")
# print(f"\ntrain_max_height: \n\t{max_height}")

# max_width = 0
# max_height = 0
# for i in range(416):
#   serialized = list(tf.data.TFRecordDataset(datasets[overlap_scores[0]]['test'][i]))[0]
#   nwu_variables = peek(serialized)
#   features, _, _, _ = read_record(serialized, nwu_variables, structured=True)
#   if features.shape[0] > max_width:
#     max_width = features.shape[0]
#   if features.shape[1] > max_height:
#     max_height = features.shape[1]
# print(f"\ntest_max_width: \n\t{max_width}")
# print(f"\ntest_max_height: \n\t{max_height}")

# dataset building function
def build_dataset(score, tau, train, validation, test, emory_variables, nwu_variables, structured=True, batch=1, prefetch=2, reads=4):
    
    #define label function for training dataset
    def threshold(value, key=score, cond=lambda x: x>=tau):
        return tf.one_hot(tf.cast(cond(value[key]), tf.int32), depth=2)

    def dataset(score, files, variables, structured, batch, prefetch, reads):    
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=reads).shuffle(len(files))
        ds = ds.map(lambda x: read_record(x, variables, structured=structured, precision=tf.float16))
        ds = ds.map(lambda x, y, z, _: (x, threshold(y, score)[0]))
        ds = ds.batch(batch).prefetch(prefetch)    
        return ds
    
    #train, validation, test datasets
    train_ds = dataset(score, train, emory_variables, structured, batch, prefetch, reads)
    validation_ds = dataset(score, validation, emory_variables, structured, batch, prefetch, reads)
    test_ds = dataset(score, test, nwu_variables, structured, batch, prefetch, reads)
    
    return train_ds, validation_ds, test_ds

# creating a moving-average model optimizer
def create_optimizer(optimizer=tf.keras.optimizers.Adam, lr=1e-5, num_updates=5):
    optimizer = optimizer(lr)
    optimizer = tfa.optimizers.MovingAverage(optimizer, num_updates=num_updates)    
    return optimizer

# generating checkpointing callbacks for moving-average models - necessary for MA to be used in training
def create_checkpoint(path, score):
    #create checkpoint for model averaging 
    ckpt_path = f'{path}{score}-cp-{{epoch:04d}}.ckpt'
    ckpt_dir = os.path.dirname(path)
    callback = tfa.callbacks.AverageModelCheckpoint(filepath=ckpt_dir, update_weights=True)
    return callback

# mask augmentation for structured tensors - sets entire tiles to zero at rate drop
def structure_mask(x, drop):
    foreground = tf.reduce_sum(tf.abs(x), 3)[0] > 0.
    foreground = tf.where(foreground)   
    total = tf.cast(tf.shape(foreground)[0], tf.float32) * (1.-drop)
    total = tf.cast(tf.math.ceil(total), tf.int32)
    select = tf.random.shuffle(foreground)[0:total, :]
    mask = tf.scatter_nd(select, 
                         tf.ones(tf.shape(select)[0], dtype=x.dtype), 
                         shape=tf.cast(tf.shape(x)[1:3], tf.int64))
    mask = tf.expand_dims(mask, axis=2)
    mask = tf.repeat(mask, tf.shape(x)[3], 2)
    mask = tf.expand_dims(mask, axis=0)
    x = x * mask
    return x

# mask augmentation for flattened tensors - sets entire tiles to zero at rate drop
def flatten_mask(x, drop):
    x = tf.random.uniform([1, 4, 2], minval=0., maxval=1.)
    order = tf.range(tf.shape(x)[1], dtype=tf.int32)
    total = tf.cast(tf.shape(x)[1], tf.float32) * (1.-drop)
    total = tf.cast(tf.math.ceil(total), tf.int32)
    select = tf.random.shuffle(order)[0:total]
    x = tf.expand_dims(tf.gather(x[0], select), axis=0)
    return x

# data augmentation for structured models, mirror-flip, rotate, additive noise
def structured_augmentation(ds, flip=True, rotate=True, drop=None, noise=None):
    if flip and tf.random.uniform(shape=[]) > 0.5:
        ds = ds.map(lambda x, y: (tf.reverse(x, axis=[1]), y))
    if flip and tf.random.uniform(shape=[]) > 0.5:
        ds = ds.map(lambda x, y: (tf.reverse(x, axis=[2]), y))
    if rotate and tf.random.uniform(shape=[]) > 0.5:
        ds = ds.map(lambda x, y: (tf.transpose(x, perm=[0, 2, 1, 3]), y))
        ds = ds.map(lambda x, y: (tf.reverse(x, axis=[2]), y))
    if rotate and tf.random.uniform(shape=[]) > 0.5:
        ds = ds.map(lambda x, y: (tf.transpose(x, perm=[0, 2, 1, 3]), y))
        ds = ds.map(lambda x, y: (tf.reverse(x, axis=[2]), y))
    if rotate and tf.random.uniform(shape=[]) > 0.5:
        ds = ds.map(lambda x, y: (tf.transpose(x, perm=[0, 2, 1, 3]), y))
        ds = ds.map(lambda x, y: (tf.reverse(x, axis=[2]), y))
    if drop is not None:
        ds = ds.map(lambda x, y: (structure_mask(x, drop), y))
    if noise is not None:
        ds = ds.map(lambda x, y: (x + tf.random.normal(shape=tf.shape(x),
                                                       stddev=noise,
                                                       dtype=x.dtype),
                                  y)
                   )
    return ds

# data augmentation for flattened models, additive noise and masking
def flattened_augmentation(ds, drop=0.05, noise=None):
    if drop > 0.:
        ds = ds.map(lambda x, y: 
                    (tf.boolean_mask(x, 
                                     tf.random.uniform([tf.shape(x)[1]],
                                                       minval=0.,
                                                       maxval=1.) > drop,
                                     axis=1),
                     y)
                   )
    if noise is not None:
        ds = ds.map(lambda x, y: (x + tf.random.normal(shape=tf.shape(x),
                                                       stddev=noise,
                                                       dtype=x.dtype),
                                  y)
                   )        
    return ds

lr = 1e-5
wd = 1e-6
dropout_rate = 0.03  # Dropout rate
label_smoothing = 0.1
patch_size = (2, 2)  # 2-by-2 sized patches
num_patch_x = 180
num_patch_y = 360
embed_dim = 64
image_dimension = 1280  # Initial image size
num_heads = 8
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window

def smooth_labels(labels, dtype, label_smoothing):
    labels = tf.cast(labels, dtype = dtype)
    labels = (1 - label_smoothing) * labels + 0.5 * label_smoothing
    return labels

class binary_focal_loss(tf.keras.losses.Loss):
    def __init__(self, alpha=[[.25, .25]], gamma=2.0, smoothing=0.1, max=1.0 - 1e-7, min=1e-7, name="Focalloss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = np.array(alpha, dtype=np.float32)
        self.gamma = gamma
        self.smoothing = smoothing
        self.max = max
        self.min = min
    
    def call(self, labels, p):
        # For numerical stability
        p = tf.clip_by_value(p, self.min, self.max)

         # Calculate Cross Entropy
        cross_entropy = -labels * K.log(p)

        # Calculate Focal Loss
        loss = self.alpha * K.pow(1 - p, self.gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))
    
    def get_config(self):
        config = {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'smoothing': self.smoothing,
            'max': self.max,
            'min': self.min,
        }
        base_config = super().get_config
        return {**base_config, **config}

def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size 
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows

def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size 
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        # print(f"\nsize of patches: \n\t{patches}")
        patch_dim = patches.shape[-1]
        patch_num1 = patches.shape[1]
        patch_num2 = patches.shape[2]
        return tf.reshape(patches, (batch_size, patch_num1 * patch_num2, patch_dim))

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

def create_vit_classifier():
    inputs = layers.Input(shape=(None, None, 1280))
    # print(f"\nsize of inputs: \n\t{inputs}")
    crop = layers.Resizing(360, 720)(inputs)
    # print(f"\nsize of crop: \n\t{crop}")
#   maxpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(crop)
#   print(f"\nsize of maxpool: \n\t{maxpool}")
  # Create features
    patches = PatchExtract(patch_size)(crop)
    # print(f"\nsize of patches: \n\t{patches}")
  # Encode patches
    encoded_patches = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(patches)
#   print(f"\nsize of encoded_patches: \n\t{encoded_patches}")
    for _ in range(3):
        x = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )(encoded_patches)
        x = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )(x)
        x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)

    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(encoded_patches)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = layers.GlobalAveragePooling1D()(x)
    # Classify outputs.
    logits = layers.Dense(2, activation='softmax', name='softmax')(x)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model
  
# create a convolutional model
def score_convolutional(score, 
                        tau, 
                        D, 
                        datasets,
                        emory_variables,
                        nwu_variables,
                        io = {'batch': 1,
                              'prefetch': 2,
                              'reads': 4},
                        augmentation = {'flip': True,
                                        'rotate': True,
                                        'drop': None,
                                        'noise': None},
                        loss='hinge',
                        optimization = {'epochs': 25,
                                        'lr': 1e-5,
                                        'num_updates': 5,
                                        'decay': 0.99},
                        metrics=[tf.keras.metrics.AUC(curve='ROC'),
                                 Balanced(threshold=0.5),
                                 Mcc(threshold=0.5),
                                 Sensitivity(threshold=0.5),
                                 Specificity(threshold=0.5)],
                        checkpoint_path = "./checkpoints/"):
    
    # create model averaging checkpoint callback - needed to update train weights with averaged
    avg_callback = create_checkpoint(checkpoint_path, score)

    # create exponential averaging optimizer
    optimizer = create_optimizer(tf.keras.optimizers.Adam, 
                                 optimization['lr'], 
                                 optimization['num_updates'])

    # create and compile model
    convolutional = create_vit_classifier()
    # print(f"Number of parameters in model: {convolutional.count_params()}")

    convolutional.compile(optimizer=optimizer,
                          loss=binary_focal_loss(),
                          metrics=metrics
                         )

    # build structured datasets for i_score
    i_train_ds, i_validation_ds, i_test_ds = build_dataset(score, tau, 
                                                           datasets[score]['train'], 
                                                           datasets[score]['validation'], 
                                                           datasets[score]['test'],
                                                           emory_variables, 
                                                           nwu_variables, 
                                                           structured=True,
                                                           batch=io['batch'], 
                                                           prefetch=io['prefetch'], 
                                                           reads=io['reads'])

    # data augmentation
    i_train_ds = structured_augmentation(i_train_ds, 
                                         augmentation['flip'],
                                         augmentation['rotate'],
                                         augmentation['noise'])

    # train model - evaluate on validation
    validation = convolutional.fit(i_train_ds,
                                   validation_data=i_validation_ds,
                                   class_weight=datasets[score]['class_weight'],
                                   batch_size=io['batch'],
                                   epochs=optimization['epochs'])
                                   # callbacks=[avg_callback]

    # re-train model on training+validation - evaluate on test
    convolutional = create_vit_classifier()
    
    # create a new exponential averaging optimizer - testing set
    optimizer = create_optimizer(tf.keras.optimizers.Adam, 
                                 optimization['lr'], 
                                 optimization['num_updates'])

    #build new model - testing set
    convolutional.compile(optimizer=optimizer,
                          loss=binary_focal_loss(),
                          metrics=metrics
                         )

    # build new structured datasets for i_score
    i_train_ds, i_validation_ds, i_test_ds = build_dataset(score, tau, 
                                                           datasets[score]['train'], 
                                                           datasets[score]['validation'], 
                                                           datasets[score]['test'],
                                                           emory_variables, 
                                                           nwu_variables, 
                                                           structured=True,
                                                           batch=io['batch'], 
                                                           prefetch=io['prefetch'], 
                                                           reads=io['reads'])

    # combine training and validation sets into a larger training set
    i_train_ds = i_train_ds.concatenate(i_validation_ds)
    
    # data augmentation
    i_train_ds = structured_augmentation(i_train_ds, 
                                         augmentation['flip'],
                                         augmentation['rotate'],
                                         augmentation['noise'])
    
    # train model - evaluate on testing
    testing = convolutional.fit(i_train_ds,
                                validation_data=i_test_ds,
                                class_weight=datasets[score]['class_weight'],
                                batch_size=io['batch'],
                                epochs=optimization['epochs'])
                                # callbacks=[avg_callback]
    
    return convolutional, validation, testing


metrics = [tf.keras.metrics.AUC(curve='ROC'),
           Balanced(threshold=0.5),
           Mcc(threshold=0.5),
           Sensitivity(threshold=0.5),
           Specificity(threshold=0.5)]

# i-score convolutional model
augmentation_conv = {'flip': True,
                     'rotate': True,
                     'drop': 0.05,
                     'noise': None}
optimization = {'epochs': 25,
                'lr': 1e-5,
                'num_updates': 5,
                'decay': 0.99}
convolutional, validation, testing = score_convolutional('i',
                                                          tau,
                                                          D,
                                                          datasets,
                                                          emory_variables,
                                                          nwu_variables,
                                                          augmentation=augmentation_conv,
                                                          optimization=optimization,
                                                          checkpoint_path = f"./checkpoints/i-transformer/")

# metrics = [tf.keras.metrics.AUC(curve='ROC'),
#            Balanced(threshold=0.5),
#            Mcc(threshold=0.5),
#            Sensitivity(threshold=0.5),
#            Specificity(threshold=0.5)]


# # t-score convolutional model
# augmentation_conv = {'flip': True,
#                      'rotate': True,
#                      'drop': 0.05,
#                      'noise': None}
# optimization = {'epochs': 20,
#                 'lr': 1e-5,
#                 'num_updates': 5,
#                 'decay': 0.99}
# convolutional, validation, testing = score_convolutional('t',
#                                                           tau,
#                                                           D,
#                                                           datasets,
#                                                           emory_variables,
#                                                           nwu_variables,
#                                                           augmentation=augmentation_conv,
#                                                           optimization=optimization,
#                                                           checkpoint_path = f"./checkpoints/t-convolutioanl/")
# %%
