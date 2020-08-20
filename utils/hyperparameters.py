DIM_LATENT = 100
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
LEARNING_RATE_DECAY = 0.5
ALPHA = 1
BETA = 0.5

# crop 178x218 image
CROP_LEFT = 40
CROP_RIGHT = 138
CROP_UPPER = 25
CROP_LOWER = 218-25

# resize
RESIZE_HEIGHT = 64
RESIZE_WIDTH = 64

# normalize
NORMALIZE_MEAN = 0.5
NORMALIZE_STDEV = 0.5


def print_hyperparameters(loss_id, save_epochs):
    print("dim latent                       : {0}".format(DIM_LATENT))
    print("number of epochs                 : {0}".format(EPOCHS))
    print("batch size                       : {0}".format(BATCH_SIZE))
    print("learning rate                    : {0}".format(LEARNING_RATE))
    print("learning rate decay              : {0}".format(LEARNING_RATE_DECAY))
    print("alpha (kl loss)                  : {0}".format(ALPHA))
    print("beta (rec loss)                  : {0}".format(BETA))
    print("rec loss                         : {0}".format(["l2_loss", "vgg123_loss",\
        "vgg345_loss"][loss_id]))
    print("crop (left, upper, right, lower) : ({0}, {1}, {2}, {3})".format(CROP_LEFT,\
        CROP_UPPER, CROP_RIGHT, CROP_LOWER))
    print("resize (height, width)           : ({0}, {1})".format(RESIZE_HEIGHT, RESIZE_WIDTH))
    print("normalize                        : x = (x - {0}) / {1}".format(
        NORMALIZE_MEAN, NORMALIZE_STDEV))
    print("save model after each epoch      : {0}".format(save_epochs))