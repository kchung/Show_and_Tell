class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # Model parameters
        self.vocab_size = 10000  # Size of vocabulary
        self.embed_dim = 512  # Word embedding dimension
        self.num_lstm_units = 512  # Number of LSTM units
        self.num_attention_heads = 8  # Number of attention heads
        self.dropout_rate = 0.5  # Dropout rate
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.clip_gradients = True
        self.max_gradient_norm = 5.0
        
        # Data parameters
        self.image_size = 224  # Input image size for VGG16
        self.max_caption_length = 50  # Maximum caption length
        self.min_word_count = 5  # Minimum word frequency for vocabulary
        
        # Special tokens
        self.start_token = 1  # Start of sentence token
        self.end_token = 2  # End of sentence token
        self.pad_token = 0  # Padding token
        self.unk_token = 3  # Unknown word token
        
        # Paths
        self.train_image_dir = 'train/images'
        self.train_caption_file = 'train/captions_train2014.json'
        self.val_image_dir = 'val/images'
        self.val_caption_file = 'val/captions_val2014.json'
        self.save_dir = 'models'
        self.log_dir = 'logs'
        
        # Save and log frequency
        self.save_period = 5  # Save model every N epochs
        self.log_period = 100  # Log every N steps
        
        # Beam search parameters
        self.beam_size = 3
        
        # CNN parameters
        self.train_cnn = False  # Whether to train the CNN
        self.cnn_learning_rate = 1e-4  # Learning rate for CNN when training
        
        # Mixed precision training
        self.use_mixed_precision = True
        self.mixed_precision_policy = 'mixed_float16'
        
        # Optimization parameters
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_epsilon = 1e-8
        self.weight_decay = 0.01
        
        # Data augmentation
        self.use_augmentation = True
        self.random_brightness = 0.2
        self.random_contrast = 0.2
        self.random_saturation = 0.2
        self.random_hue = 0.1
        self.random_flip = True
        
        # Early stopping
        self.use_early_stopping = True
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 0.001
        
        # Regularization
        self.l2_reg = 0.01
        self.label_smoothing = 0.1

        # about the model architecture
        self.cnn = 'vgg16'               # 'vgg16' or 'resnet50'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_initalize_layers = 1 ## Changed from 2 to 1    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 1    ## Changed from 2 to 1   # 1 or 2
        self.dim_decode_layer = 1024

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_steps_per_decay = 100000
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 1000
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'
        self.vocabulary_size = 5000

        # about the training
        self.temp_annotation_file = './train/anns.csv'
        self.temp_data_file = './train/data.npy'

        # about the evaluation
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = False

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'

        self.trainable_variable = False
