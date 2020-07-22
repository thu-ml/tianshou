#!/usr/bin/env python3

import os
import shutil
from subprocess import Popen, PIPE


def main():
    if os.path.exists('_build'):
        shutil.rmtree('_build')
    p = Popen(['make', 'html'], stderr=PIPE)
    _, err = p.communicate()
    err = [i for i in err.decode().splitlines() if 'WARNING' in i]
    assert len(err) == 0, '\n'.join(err)


if __name__ == '__main__':
    main()
