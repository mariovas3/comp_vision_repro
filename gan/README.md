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