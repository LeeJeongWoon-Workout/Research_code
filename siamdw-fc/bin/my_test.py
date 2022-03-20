from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import SiamFCTracker

import argparse

if __name__ == '__main__':


    tracker = SiamFCTracker('/home/airlab/PycharmProjects/pythonProject5/SiamDW-FC/bin/models/siamfcres22_1.pth') #初始化一个追踪器

    e = ExperimentLaSOT(root_dir='/home/airlab/PycharmProjects/pythonProject5/data/LaSOT')
    e.run(tracker,visualize=True)
    e.report([tracker.name])
