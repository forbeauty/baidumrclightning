import os
import argparse
import yaml


class YamlInputAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if os.path.exists(os.path.join(os.getcwd(), 'configs', values)):
            try:
                with open(os.path.join(os.getcwd(), 'configs', values)) as fp:
                    raw = fp.read()
            except Exception:
                raise argparse.ArgumentError(self, 'invalid yaml file')
        else:
            raw = values
        try:
            v = yaml.safe_load(raw)
            if not isinstance(v, dict):
                raise argparse.ArgumentError(
                    self, 'input file is not a dictionary'
                )
            setattr(namespace, self.dest, v)
        except ValueError:
            raise argparse.ArgumentError(self, 'invalid yaml content')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        action=YamlInputAction,
        help='description options'
    )
    # parser.add_argument(
    #     '--config',
    #     default='configs/config.yml',
    #     help='Path to the configration yaml file.'
    # )
    parser.add_argument(
        '--save_path',
        default='run1',
        help='Path output.'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        nargs="+",
        default=None,
        help="number of gpu to use"
    )
    parser.add_argument(
        '--tpu',
        type=int,
        default=None,
        help="number of tpu to use"
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit model on a few examples, meant for debugging."
    )

    parser.add_argument(
        "--fold",
        default=0,
        type=int,
        help="n-th fold."
    )
    return parser.parse_args()
