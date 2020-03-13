#!/bin/sh
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics                 
flake8 . --count --exit-zero --max-complexity=20 --max-line-length=79 --statistics
