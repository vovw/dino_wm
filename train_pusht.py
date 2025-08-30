#!/usr/bin/env python3
"""
Simple script to train DINO-WM on PushT environment.
"""
import os
import sys
from train import main

if __name__ == "__main__":
    # Override sys.argv to set the environment
    sys.argv = ["train_pusht.py", "--env", "pusht"] + sys.argv[1:]
    main()
