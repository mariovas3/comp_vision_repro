## Short summary:
* I used the DCGAN architecture - conv transpose modules in generator and conv modules in discriminator.
* This is much more brittle than trianing VAE. In particular, trading off how often to train generator relative to discriminator.
* Should probably do a larger hparam sweep at some point. Neural architecture search might also help.

## Running experiments:

* First navigate to root of repo (one level above this dir).
* Then export the `PYTHONPATH`:

    ```bash
    export PYTHONPATH=.
    ```
* Then set wandb environment variable to avoid `BrokenPipeError` at end of run:

    ```bash
    export WANDB_START_METHOD="thread"
    ```
* Then run the training reading the config from `train_mnist_gan.yaml`:

    ```bash
    python gan/run_experiment.py fit --config train_mnist_gan.yaml
    ```