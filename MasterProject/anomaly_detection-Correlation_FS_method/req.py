# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:31:57 2024

@author: eriks
"""

import subprocess
import sys

def install(name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', name]) 
with open('requirements.txt') as f:
    lines = f.readlines()    
    for x in lines:
        print(f"Installing {x}....")
        install(x)
