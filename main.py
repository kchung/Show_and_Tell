#!/usr/bin/python
import tensorflow as tf
import argparse
from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
from PIL import Image
import numpy as np
from imagenet_classes import class_names

def parse_args():
    parser = argparse.ArgumentParser(description='Show and Tell: Neural Image Caption Generator')
    parser.add_argument('--phase', type=str, default='train',
                        help='The phase can be train, eval or test')
    parser.add_argument('--load', action='store_true',
                        help='Turn on to load a pretrained model')
    parser.add_argument('--model_file', type=str,
                        help='If specified, load a pretrained model from this file')
    parser.add_argument('--load_cnn', action='store_true',
                        help='Turn on to load a pretrained CNN model')
    parser.add_argument('--cnn_model_file', type=str, default='./vgg16_no_fc.npy',
                        help='The file containing a pretrained CNN model')
    parser.add_argument('--train_cnn', action='store_true',
                        help='Turn on to train both CNN and RNN')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='The size of beam search for caption generation')
    parser.add_argument('--image_file', type=str, default='./man.jpg',
                        help='The file to test the CNN')
    return parser.parse_args()

def main():
    # Enable mixed precision training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    args = parse_args()
    config = Config()
    config.phase = args.phase
    config.train_cnn = args.train_cnn
    config.beam_size = args.beam_size
    config.trainable_variable = args.train_cnn

    if args.phase == 'train':
        # training phase
        data = prepare_train_data(config)
        model = CaptionGenerator(config)
        
        if args.load:
            model.load(args.model_file)
        
        if args.load_cnn:
            model.load_cnn(args.cnn_model_file)
            
        model.train(data)

    elif args.phase == 'eval':
        # evaluation phase
        coco, data, vocabulary = prepare_eval_data(config)
        model = CaptionGenerator(config)
        model.load(args.model_file)
        model.eval(coco, data, vocabulary)

    elif args.phase == 'test_loaded_cnn':
        # testing only cnn
        model = CaptionGenerator(config)
        
        if args.load_cnn:
            model.load_cnn(args.cnn_model_file)

        img = Image.open(args.image_file)
        img = img.resize((224, 224))
        img = np.array(img)
        
        if img.shape[-1] == 4:  # Remove alpha channel if present
            img = img[..., :3]
            
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.expand_dims(img, 0)  # Add batch dimension
        
        probs = model.test_cnn(img)
        preds = tf.argsort(probs[0], direction='DESCENDING')[:5]
        
        for p in preds:
            print(f"{class_names[p.numpy()]}: {probs[0][p].numpy():.4f}")

    else:
        # testing phase
        data, vocabulary = prepare_test_data(config)
        model = CaptionGenerator(config)
        model.load(args.model_file)
        model.test(data, vocabulary)

if __name__ == '__main__':
    main()
