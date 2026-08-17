# Methodology

The project loads three RGB images per sample, applies synchronized training augmentation, normalizes each image, and concatenates the three tensors into a 9-channel input.

The custom CNN contains six convolutional blocks with channel widths 16, 32, 64, 96, 128, and 192. Each block uses a custom convolution, batch normalization, ReLU, and max-pooling layer. Adaptive average pooling reduces the final feature map to 1×1, followed by scratch linear layers and dropout before the 16-class output.

Training uses cross-entropy loss with label smoothing, Adam optimization, a ReduceLROnPlateau scheduler, and Macro F1 for model selection. The supplied notebook trains for up to 80 epochs and saves the best checkpoint based on validation Macro F1.
