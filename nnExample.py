#!/usr/bin/env python3
import sys, math, csv
import numpy as np
import matplotlib.pyplot as plot

def main():
    a = np.zeros([3,2])
    a[1,1] = 2
    a[2,1] = 4
    print(a)


if __name__ == '__main__':
    main()
