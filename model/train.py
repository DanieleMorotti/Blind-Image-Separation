from utils import check_training_args
from data import create_generators
from BIS_model import build_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


def initialize_callbacks(monitor, patience_es, min_delta_es, patience_lr, factor_lr, min_delta_lr, mode_lr, path_cp):
    
    early_stopping = EarlyStopping(monitor=monitor, patience=patience_es, min_delta=min_delta_es)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=factor_lr, patience=patience_lr, min_delta=min_delta_lr, mode=mode_lr)
    model_cp = ModelCheckpoint(path_cp, monitor=monitor, verbose=1, save_weights_only=True, save_best_only=True)

    return [early_stopping, reduce_lr, model_cp]


# Train the model with the desired parameters and return the history
def train_model(
    model, train_gen, val_gen, lr=1e-3, loss='mse', metrics=['mean_squared_error'], callbacks=[], 
    train_steps=6500, val_steps=500, epochs=200
):

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=metrics)
    print("Start training...")
    history = model.fit(
        x=train_gen, epochs=epochs, validation_data=val_gen, validation_steps=val_steps, 
        steps_per_epoch=train_steps, callbacks=callbacks
    )
    print("Training terminated.\n")
    return history, metrics


# Resume training starting from the weights in "start_weights"
def resume_training(
    model, start_weights, train_gen, val_gen, lr=1e-3, loss='mse', metrics=['mean_squared_error'], callbacks=[], 
    train_steps=6500, val_steps=500, epochs=200
):
    print(f"Loading weights from {start_weights}")
    model.load_weights(start_weights)

    history = train_model(
        model, train_gen, val_gen, lr=lr, loss=loss, metrics=metrics, 
        callbacks=callbacks, train_steps=train_steps, val_steps=val_steps, epochs=epochs
    )

    return history, metrics


# Actually run the training after creating the dataset and a model
def execute_training(model, mode, start_weights, weights_cp_path):
    
    train_gen, val_gen, _ = create_generators(batch_train=32, batch_val=32, batch_test=1024)
    callbacks = initialize_callbacks("val_loss", 8, 1e-6, 4, 0.4, 1e-6, "min", weights_cp_path)

    print("TRAINING\n")
    if mode == 'new_train':
        history, metrics = train_model(
                                model, train_gen, val_gen, lr=1e-3, loss='mse', metrics=['mean_squared_error'], 
                                callbacks=callbacks, train_steps=6500, val_steps=500, epochs=200
                            )
    else:
        history, metrics = resume_training(
                                model, start_weights, train_gen, val_gen, lr=1e-3, loss='mse', metrics=['mean_squared_error'],
                                callbacks=callbacks, train_steps=6500, val_steps=500, epochs=200
                            )
    return history, metrics


if __name__ == '__main__':
        
    model = build_model(input_shape=(32, 32, 1), n_ch=128, blocks=3, conv_per_b=2)
    mode, start_weights = check_training_args()
    execute_training(model, mode, start_weights, "../weights/cp_weights.h5")
   

