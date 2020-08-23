# arguments.py
class args:
    ROOT_PATH = "/fred/oz138/test/data/train_audio"
    MODEL_PATH = "/fred/oz138/test/model"

    num_classes = 264
    max_duration = 5  # seconds

    sample_rate = 32000

    img_height = 128
    img_width = 313

    batch_size = 250
    num_workers = 4
    epochs = 15

    lr = 0.0009
    wd = 1e-5
    momentum = 0.9
    eps = 1e-8
    betas = (0.9, 0.999)

    melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 20,
        "fmax": 16000
    }

