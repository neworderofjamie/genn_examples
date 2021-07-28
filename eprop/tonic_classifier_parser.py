from argparse import ArgumentParser

def parse_arguments(parent_parser=None, description=None):
    # Build command line parse
    if parent_parser is None:
        parser = ArgumentParser(description=description)
    else:
        parser = ArgumentParser(description=description, parents=[parent_parser])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--feedforward", action="store_true")
    parser.add_argument("--num-recurrent-alif", type=int, default=256)
    parser.add_argument("--num-recurrent-lif", type=int, default=0)
    parser.add_argument("--dataset", choices=["smnist", "shd"], required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    # Determine output directory name and create if it doesn't exist
    name_suffix = "%u%s%s%s%s" % (args.num_recurrent_alif, 
                                  "_%u" % args.num_recurrent_lif if args.num_recurrent_lif > 0 else "",
                                  "_feedforward" if args.feedforward else "", 
                                  "_%.1f" % args.dt if args.dt != 1.0 else "",
                                  args.suffix)
    output_directory = "%s_%s" % (args.dataset, name_suffix)

    return name_suffix, output_directory, args
