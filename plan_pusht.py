#!/usr/bin/env python3
"""
Simple script to run planning with DINO-WM on PushT environment.
"""
import os
import sys
from plan import main

if __name__ == "__main__":
    # Override sys.argv to set the environment
    sys.argv = ["plan_pusht.py", "--env", "pusht"] + sys.argv[1:]
    main()
