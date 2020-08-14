EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.5
ALPHA = 1
BETA = 0.5

def print_hyperparameters():
    print("number of epochs     : {0}".format(EPOCHS))
    print("batch size           : {0}".format(BATCH_SIZE))
    print("learning rate        : {0}".format(LEARNING_RATE))
    print("weight decay         : {0}".format(WEIGHT_DECAY))
    print("alpha (kl loss)      : {0}".format(ALPHA))
    print("beta (rec loss)      : {0}".format(BETA))