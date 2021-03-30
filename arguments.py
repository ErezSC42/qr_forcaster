import argparse


def get_params():
    parser = argparse.ArgumentParser(description="PyTorch ForecasterQR")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--use-nni", action="store_true", default=False)
    parser.add_argument("--use-forking-sequences", action="store_true", default=False)
    parser.add_argument("--max-sequence-len", type=int, default=168)
    parser.add_argument("--forcast-horizons", type=int, default=24)
    parser.add_argument("--encoder-layer-count", type=int, default=1)
    parser.add_argument("--encoder-hidden-dim", type=int, default=32)
    parser.add_argument("--decoder-context-dim", type=int, default=32)
    parser.add_argument("--dataset-num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)

    args, _ = parser.parse_known_args()
    return args