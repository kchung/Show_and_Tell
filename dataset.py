import tensorflow as tf
import json
import os
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm

from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary

class DataSet(object):
    def __init__(self,
                 image_ids,
                 image_files,
                 batch_size,
                 word_idxs=None,
                 masks=None,
                 is_train=False,
                 shuffle=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.word_idxs = np.array(word_idxs)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))

        image_files = self.image_files[current_idxs]
        if self.is_train:
            word_idxs = self.word_idxs[current_idxs]
            masks = self.masks[current_idxs]
            self.current_idx += self.batch_size
            return image_files, word_idxs, masks
        else:
            self.current_idx += self.batch_size
            return image_files

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count

def load_coco_data(config, split='train'):
    """Load COCO dataset."""
    if split == 'train':
        image_dir = config.train_image_dir
        caption_file = config.train_caption_file
    else:
        image_dir = config.val_image_dir
        caption_file = config.val_caption_file
    
    # Load captions
    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    # Process captions and build vocabulary
    captions = []
    image_ids = []
    
    for annotation in data['annotations']:
        captions.append(annotation['caption'].lower())
        image_ids.append(annotation['image_id'])
    
    # Build vocabulary
    if split == 'train':
        word_counts = Counter()
        for caption in captions:
            word_counts.update(caption.split())
        
        vocab = ['<pad>', '<start>', '<end>', '<unk>']
        vocab.extend([word for word, count in word_counts.items()
                     if count >= config.min_word_count])
        
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        
        config.vocab_size = len(vocab)
        return create_dataset(config, image_dir, captions, image_ids, word_to_idx), word_to_idx, idx_to_word
    else:
        return create_dataset(config, image_dir, captions, image_ids, None)

def preprocess_image(image_path, config):
    """Load and preprocess image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [config.image_size, config.image_size])
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

def preprocess_caption(caption, word_to_idx, config):
    """Preprocess caption."""
    words = caption.split()
    if len(words) > config.max_caption_length - 2:  # -2 for start and end tokens
        words = words[:config.max_caption_length - 2]
    
    caption = []
    caption.append(word_to_idx['<start>'])
    caption.extend([word_to_idx.get(word, word_to_idx['<unk>']) for word in words])
    caption.append(word_to_idx['<end>'])
    
    # Pad caption
    while len(caption) < config.max_caption_length:
        caption.append(word_to_idx['<pad>'])
    
    return caption

def create_dataset(config, image_dir, captions, image_ids, word_to_idx=None):
    """Create TensorFlow dataset."""
    image_paths = [os.path.join(image_dir, f'COCO_{image_id:012d}.jpg')
                  for image_id in image_ids]
    
    if word_to_idx is not None:
        # Training dataset
        captions = [preprocess_caption(caption, word_to_idx, config)
                   for caption in captions]
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
        
        # Shuffle and batch
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(
            lambda img_path, cap: (preprocess_image(img_path, config), cap),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if config.use_augmentation:
            dataset = dataset.map(
                augment_data,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # Validation/Test dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
        dataset = dataset.map(
            lambda img_path, cap: (preprocess_image(img_path, config), cap),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

@tf.function
def augment_data(image, caption):
    """Apply data augmentation to images."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    return image, caption

def prepare_train_data(config):
    """Prepare training data."""
    print("Preparing training data...")
    dataset, word_to_idx, idx_to_word = load_coco_data(config, split='train')
    config.word_to_idx = word_to_idx
    config.idx_to_word = idx_to_word
    return dataset

def prepare_eval_data(config):
    """Prepare validation data."""
    print("Preparing validation data...")
    dataset = load_coco_data(config, split='val')
    return dataset

def prepare_test_data(config):
    """Prepare test data."""
    print("Preparing test data...")
    # Create dataset from test directory
    test_image_dir = 'test/images'
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    image_paths = [os.path.join(test_image_dir, img) for img in test_images]
    dummy_captions = [''] * len(image_paths)  # Dummy captions for consistency
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, dummy_captions))
    dataset = dataset.map(
        lambda img_path, cap: (preprocess_image(img_path, config), cap),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.build(coco.all_captions())
    vocabulary.save(config.vocabulary_file)
    return vocabulary
