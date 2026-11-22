import tensorflow as tf

def bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1, gamma):
    """Calculate the loss for the BunDLe Net

    Args:
        yt1_upper: Output from the upper arm of the BunDLe Net.
        yt1_lower: Output from the lower arm of the BunDLe Net.
        bt1_upper: Predicted output from the upper arm of the BunDLe Net.
        b_train_1: True output for training.
        gamma (float): Tunable weight for the DCC loss component.

    Returns:
        tuple: A tuple containing the DCC loss, behavior loss, and total loss.
    """
    mse = tf.keras.losses.MeanSquaredError()
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    DCC_loss = mse(yt1_upper, yt1_lower)
    behaviour_loss = scce(b_train_1, bt1_upper)
    total_loss = gamma * DCC_loss + (1 - gamma) * behaviour_loss
    return gamma * DCC_loss, (1 - gamma) * behaviour_loss, total_loss