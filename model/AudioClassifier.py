import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras import layers, models, callbacks, regularizers


def build_cnn_model(
    input_shape,
    num_classes,
    spatial_dropout_rate_1=0.07,
    spatial_dropout_rate_2=0.14,
    l2_rate=0.0005,
):
    """
    Build CNN model
    Args:
        input_shape: Shape of input MFCC features
        num_classes: Number of classes to predict

    """
    model = models.Sequential(
        [
            layers.Input(input_shape),
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                kernel_regularizer=regularizers.l2(l2_rate),
            ),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(spatial_dropout_rate_1),
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                kernel_regularizer=regularizers.l2(l2_rate),
            ),
            layers.LeakyReLU(alpha=0.1),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.SpatialDropout2D(spatial_dropout_rate_1),
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                kernel_regularizer=regularizers.l2(l2_rate),
            ),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(spatial_dropout_rate_2),
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                kernel_regularizer=regularizers.l2(l2_rate),
            ),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_ann_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(input_shape),
            # Frist Block
            layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.PReLU(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            # Second Blcok
            layers.Dense(256, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            # Third dense block
            layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.PReLU(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Fourth Blcok
            layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train(model, X, y, model_name, batch_size=64, epochs=60, patience=10):
    """
    Train the model
    Args:
        X: Training features
        y: Training labels
        model_name: Name of the model that's going to be saved
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience

    Returns:
        history: collection that contains training accuracy
    """
    # Create callbacks
    checkpoint = callbacks.ModelCheckpoint(
        f"{model_name}_best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Learning rate scheduler
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    # Tensorboard
    log_dir = f"./logs/fit/{model_name}" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train model
    history = model.fit(
        x=X,
        y=y,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tensorboard_callback, checkpoint, early_stopping, reduce_lr],
    )

    return history


def plot_train_history(history, x_ticks_vertical=False):
    history = history.history

    # min loss / max accs
    min_loss = min(history["loss"])
    min_val_loss = min(history["val_loss"])
    max_accuracy = max(history["accuracy"])
    max_val_accuracy = max(history["val_accuracy"])

    # x pos for loss / acc min/max
    min_loss_x = history["loss"].index(min_loss)
    min_val_loss_x = history["val_loss"].index(min_val_loss)
    max_accuracy_x = history["accuracy"].index(max_accuracy)
    max_val_accuracy_x = history["val_accuracy"].index(max_val_accuracy)

    # summarize history for loss, display min
    plt.figure(figsize=(16, 8))
    plt.plot(history["loss"], color="#1f77b4", alpha=0.7)
    plt.plot(history["val_loss"], color="#ff7f0e", linestyle="--")
    plt.plot(
        min_loss_x,
        min_loss,
        marker="o",
        markersize=3,
        color="#1f77b4",
        alpha=0.7,
        label="Inline label",
    )
    plt.plot(
        min_val_loss_x,
        min_val_loss,
        marker="o",
        markersize=3,
        color="#ff7f0e",
        alpha=0.7,
        label="Inline label",
    )
    plt.title("Model loss", fontsize=20)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(
        ["Train", "Test", ("%.3f" % min_loss), ("%.3f" % min_val_loss)],
        loc="upper right",
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        borderpad=1,
    )

    if x_ticks_vertical:
        plt.xticks(np.arange(0, len(history["loss"]), 5.0), rotation="vertical")
    else:
        plt.xticks(np.arange(0, len(history["loss"]), 5.0))

    plt.show()

    # summarize history for accuracy, display max
    plt.figure(figsize=(16, 6))
    plt.plot(history["accuracy"], alpha=0.7)
    plt.plot(history["val_accuracy"], linestyle="--")
    plt.plot(
        max_accuracy_x,
        max_accuracy,
        marker="o",
        markersize=3,
        color="#1f77b4",
        alpha=0.7,
    )
    plt.plot(
        max_val_accuracy_x,
        max_val_accuracy,
        marker="o",
        markersize=3,
        color="orange",
        alpha=0.7,
    )
    plt.title("Model accuracy", fontsize=20)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(
        ["Train", "Test", ("%.2f" % max_accuracy), ("%.2f" % max_val_accuracy)],
        loc="upper left",
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        borderpad=1,
    )
    plt.figure(num=1, figsize=(10, 6))

    if x_ticks_vertical:
        plt.xticks(np.arange(0, len(history["accuracy"]), 5.0), rotation="vertical")
    else:
        plt.xticks(np.arange(0, len(history["accuracy"]), 5.0))

    plt.show()


def plot_confusion_matrix(
    cm, classes, normalized=False, title=None, cmap=plt.cm.Blues, size=(10, 10)
):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalized else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()
