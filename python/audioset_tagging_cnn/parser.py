import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Example of parser. ')

    parser.add_argument("yaml", help="yaml file")

    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=160)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=8000)

    args = parser.parse_args()
    return args
