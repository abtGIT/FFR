#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:25:45 2020

@author: dai
"""
from argparse import ArgumentParser
import shutil

def is_valid_path(parser, arg):
    if not shutil.os.path.exists(arg):
        parser.error("The path %s does not exist!" % arg)


parser = ArgumentParser(description="parser for various directory paths")
parser.add_argument("--base_dir", help="base directory path",
                    dest='base')
parser.add_argument('--folders', help= ' source, test and train directories path',
                    nargs='+', dest='folders')
parser.add_argument('--aug_dirs', help= ' directories path for augmentation',
                    nargs='+', dest='augdir')
args = vars(parser.parse_args())
print(args["base"])
print(tuple(args["folders"]))
print(args['augdir'])