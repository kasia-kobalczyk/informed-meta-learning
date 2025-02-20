import argparse
import toml


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    @classmethod
    def from_toml(cls, file_path):
        with open(file_path) as f:
            config_dict = toml.load(f)
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

    def write_config(self, file_path):
        with open(file_path, "w") as f:
            toml.dump(self.__dict__, f)

    def get(self, item):
        return self.__dict__.get(item, None)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name", type=str, help="Project name", default="meta-regression"
    )
    # training
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--load-dir", type=str, help="Load directory")
    parser.add_argument("--load-it", type=str, help="Load iteration", default="best")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("--num-epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument(
        "--sort-context",
        type=str2bool,
        const=True,
        nargs="?",
        help="Sort context",
        default=False,
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--decay-lr", type=int, help="Decay learning rate", default=10)
    parser.add_argument("--train-split", type=str, help="Train split", default="train")
    parser.add_argument("--val-split", type=str, help="Validation split", default="val")
    parser.add_argument(
        "--n-trials", type=int, help="Number of optuna trials", default=1
    )
    parser.add_argument("--beta", type=float, help="Beta VAE", default=1)

    # dataloader
    parser.add_argument(
        "--dataset", type=str, help="Dataset", default="custom-regression"
    )
    parser.add_argument("--split-file", type=str, help="Split file", default=None)
    parser.add_argument(
        "--knowledge-type", type=str, help="Knowledge type", default="none"
    )
    parser.add_argument(
        "--min-num-context", type=int, help="Minimum number of context", default=0
    )
    parser.add_argument(
        "--max-num-context", type=int, help="Maximum number of context", default=100
    )
    parser.add_argument(
        "--num-targets", type=int, help="Number of targets", default=100
    )
    parser.add_argument("--noise", type=float, help="Observation noise std", default=0)
    parser.add_argument("--x-sampler", type=str, help="X sampler", default="uniform")

    # knowledge and parameter freezing
    parser.add_argument(
        "--use-knowledge",
        type=str2bool,
        const=True,
        nargs="?",
        help="Use text inputs",
        default=False,
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        help="Text encoder",
        default="none",
        choices=["simple", "none", "roberta", "set", "set2", "mlp"],
    )
    parser.add_argument(
        "--freeze-llm",
        type=str2bool,
        const=True,
        nargs="?",
        help="Freeze LLM",
        default=True,
    )
    parser.add_argument(
        "--tune-llm-layer-norms",
        type=str2bool,
        const=True,
        nargs="?",
        help="Tune LLM layer norms",
        default=False,
    )
    parser.add_argument(
        "--train-num-z-samples",
        type=int,
        help="Number of training z samples",
        default=1,
    )
    parser.add_argument(
        "--test-num-z-samples", type=int, help="Number of testing z samples", default=16
    )
    parser.add_argument(
        "--knowledge-dropout", type=float, help="Knowledge dropout", default=0.3
    )

    # model architecture
    parser.add_argument("--input-dim", type=int, help="Input dimension", default=1)
    parser.add_argument("--output-dim", type=int, help="Output dimension", default=1)
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension", default=128)
    parser.add_argument(
        "--xy-encoder-num-hidden",
        type=int,
        help="Number of XY encoder hidden layers",
        default=2,
    )
    parser.add_argument(
        "--xy-encoder-hidden-dim",
        type=int,
        help="XY encoder hidden ndimension size",
        default=None,
    )
    parser.add_argument(
        "--xy-self-attention",
        type=str,
        help="XY self attention",
        default="none",
        choices=["none", "dot", "multihead"],
    )
    parser.add_argument(
        "--xy-self-attention-num-layers",
        type=int,
        help="XY self attention number of layers",
        default=1,
    )
    parser.add_argument(
        "--data-agg-func",
        type=str,
        help="Data aggregation function",
        default="mean",
        choices=["mean", "sum", "none", "cross-attention"],
    )
    parser.add_argument(
        "--latent-encoder-num-hidden",
        type=int,
        help="Number of latent encoder hidden layers",
        default=1,
    )
    parser.add_argument(
        "--decoder-hidden-dim", type=int, help="Decoder hidden dimension", default=None
    )
    parser.add_argument(
        "--decoder-num-hidden",
        type=int,
        help="Number of decoder hidden layers",
        default=3,
    )
    parser.add_argument(
        "--decoder-activation", type=str, help="Decoder activation", default="gelu"
    )
    parser.add_argument(
        "--x-transf-dim", type=int, help="X transformation dimension", default=None
    )
    parser.add_argument(
        "--x-encoder-num-hidden",
        type=int,
        help="Number of X encoder hidden layers",
        default=1,
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path",
        default="latent",
        choices=["latent", "deterministic", "both"],
    )
    parser.add_argument(
        "--knowledge-extractor-num-hidden",
        type=int,
        help="Number of knowledge extractor hidden layers",
        default=2,
    )
    parser.add_argument(
        "--knowledge-extractor-hidden-dim",
        type=int,
        help="Knowledge extractor hidden dimension",
        default=None,
    )
    parser.add_argument(
        "--knowledge-merge",
        type=str,
        help="Knowledge merge",
        default="concat",
        choices=["concat", "sum", "mlp"],
    )
    parser.add_argument(
        "--knowledge-dim",
        type=int,
        help="Dimension of knowledge representaiton",
        default=None,
    )
    # saving args
    parser.add_argument(
        "--run-name-prefix", type=str, help="Run name prefix", default="run"
    )
    parser.add_argument(
        "--run-name-suffix", type=str, help="Run name suffix", default="tuned"
    )

    # Add other arguments as needed

    args = parser.parse_args()

    if args.xy_encoder_hidden_dim is None:
        args.xy_encoder_hidden_dim = args.hidden_dim * 3
    if args.decoder_hidden_dim is None:
        args.decoder_hidden_dim = args.hidden_dim
    if args.x_transf_dim is None:
        args.x_transf_dim = args.hidden_dim
    if args.knowledge_extractor_hidden_dim is None:
        args.knowledge_extractor_hidden_dim = args.hidden_dim
    if args.knowledge_dim is None:
        args.knowledge_dim = args.hidden_dim

    print("Setting config.toml")
    config = Config.from_args(args)

    config.write_config("config.toml")

    return config


if __name__ == "__main__":
    config = main()
