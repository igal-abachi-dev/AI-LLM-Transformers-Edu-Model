"""Thin command wrapper for the canonical pretraining entry point."""

# STEP 3 OF THE PIPELINE. This file is only an alias -- the real entry point, with
# all the arguments and the actual loop, is `train/pretrain.py`. Start there.

from train.pretrain import main

if __name__ == "__main__":
    main()
