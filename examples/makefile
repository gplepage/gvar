# Copyright (c) 2015 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

PYTHON = python

run:
	$(PYTHON) pendulum-clock.py > pendulum-clock.tmp
	- diff -w -I svdcut -B pendulum-clock.out pendulum-clock.tmp
	$(PYTHON) svdcut.py > svdcut.tmp
	- diff -w -I svdcut -B svdcut.out svdcut.tmp
	$(PYTHON) integrate.py > integrate.tmp
	- diff -w -I svdcut -B integrate.out integrate.tmp

update-all:
	$(PYTHON) pendulum-clock.py > pendulum-clock.out
	$(PYTHON) svdcut.py > svdcut.out
	$(PYTHON) integrate.py > integrate.out

clean:
	rm -f *.pyc *.tmp *.p *.prof

