#/bin/bash
find . -name '*.pyc' | xargs rm
find . -name '.??*' | xargs rm
find . -name '*.dat' | xargs rm
