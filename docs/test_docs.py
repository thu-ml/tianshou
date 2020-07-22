#!/usr/bin/env python3

import os
import shutil
from subprocess import Popen, PIPE

latex_warning = "WARNING: LaTeX command 'latex' cannot be run (needed for \
math display), check the imgmath_latex setting"


def main():
    if os.path.exists('_build'):
        shutil.rmtree('_build')
    p = Popen(['make', 'html'], stderr=PIPE)
    _, err = p.communicate()
    err = [i for i in err.decode().splitlines()
           if 'WARNING' in i and latex_warning not in i]
    assert len(err) == 0, '\n'.join(err)


if __name__ == '__main__':
    main()
