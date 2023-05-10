# Created by G. Peter Lepage (Cornell University) on 2008-02-12.
# Copyright (c) 2008-2020 G. Peter Lepage.
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

PIP = python -m pip
PYTHON = python
PYTHONVERSION = python`python -c 'import platform; print(platform.python_version())'`
VERSION = `python -c 'import gvar; print (gvar.__version__)'`

DOCFILES :=  $(shell ls doc/source/conf.py doc/source/*.{rst,png})
SRCFILES := $(shell ls setup.py src/gvar/*.{py,pyx})
CYTHONFILES := src/gvar/_bufferdict.c src/gvar/_gvarcore.c src/gvar/_svec_smat.c src/gvar/_utilities.c src/gvar/dataset.c

install-user : 
	$(PIP) install . --user --no-cache-dir

install install-sys : 
	$(PIP) install . --no-cache-dir

uninstall :			# mostly works (may leave some empty directories)
	$(PIP) uninstall gvar

update :
	make uninstall install 

.PHONY : doc 

doc-html doc:
	make doc/html/index.html

doc/html/index.html : $(SRCFILES) $(DOCFILES) setup.cfg
	sphinx-build -b html doc/source doc/html

clear-doc:
	rm -rf doc/html
	
doc-zip doc.zip:
	cd doc/html; zip -r doc *; mv doc.zip ../..


sdist: $(SRCFILES) # source distribution
	$(PYTHON) -m build --sdist

.PHONY: tests

tests test-all:
	@echo 'N.B. Some tests involve random numbers and so fail occasionally'
	@echo '     (less than 1 in 100 times) due to multi-sigma fluctuations.'
	@echo '     Run again if any test fails.'
	@echo ''
	$(PYTHON) -m unittest discover

run run-examples:
	$(MAKE) -C examples PYTHON=$(PYTHON) run

register-pypi:
	python setup.py register # use only once, first time

upload-twine: $(CYTHONFILES)
	twine upload dist/gvar-$(VERSION).tar.gz

upload-git: $(CYTHONFILES)
	echo  "version $(VERSION)"
	make doc-html
	git diff --exit-code
	git diff --cached --exit-code
	git push origin master

tag-git:
	echo  "version $(VERSION)"
	git tag -a v$(VERSION) -m "version $(VERSION)"
	git push origin v$(VERSION)

test-download:
	-$(PIP) uninstall gvar
	$(PIP) install gvar --no-cache-dir

test-readme:
	python setup.py --long-description | rst2html.py > README.html

clean:
	rm -f -r build
	rm -rf __pycache__
	rm -rf src/*.egg-info
	rm -f *.so *.tmp *.pyc *.prof .coverage doc.zip
	rm -f -r dist/*
	rm -f -r doc/build
	$(MAKE) -C doc/source clean
	$(MAKE) -C tests clean
	$(MAKE) -C examples clean


