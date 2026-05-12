#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    proc_utils.py
    Utility functions for running subprocesses with limited parallelism.
'''

import sys
import subprocess
import time

def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

def wait_for_slot(procs, max_parallel):
    while len(procs) >= max_parallel:
        for p in procs[:]:
            ret = p.poll()
            if ret is not None:
                procs.remove(p)
                if ret != 0:
                    sys.exit(ret)
        time.sleep(1)

def wait_all(procs):
    for p in procs:
        if p.wait() != 0:
            sys.exit(1)
    procs.clear()