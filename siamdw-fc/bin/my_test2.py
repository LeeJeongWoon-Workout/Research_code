from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import SiamFCTracker

import argparse

if __name__ == '__main__':


    tracker = SiamFCTracker('/home/airlab/PycharmProjects/pythonProject5/SiamDW-FC/bin/models/siamfcres22_2.pth') #初始化一个追踪器

    e = ExperimentGOT10k('/home/airlab/PycharmProjects/pythonProject5/data/GOT-10K', subset='test')

    e.run(tracker,visualize=True)
    e.report([tracker.name])

