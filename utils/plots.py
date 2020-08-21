import matplotlib.pyplot as plt


def grid_add_img(img, fig, rows, cols, pos):
    fig.add_subplot(rows, cols, pos)
    plt.imshow(img)
    plt.axis('off')