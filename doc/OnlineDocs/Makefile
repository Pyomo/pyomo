# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Pyomo
SOURCEDIR     = .
BUILDDIR      = _build
APIDIR        = api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Removing *.spy"
	@find . -name \*.spy -delete
	@for D in $(BUILDDIR) $(SOURCEDIR)/$(APIDIR); do \
	    if test -d "$$D"; then echo "Removing $$D"; rm -r "$$D"; fi \
	 done
	@for D in ext src; do \
	    if test -d "$$D/__pycache__"; then \
                echo "Removing $$D/__pycache__"; rm -r "$$D/__pycache__" ; \
            fi \
	 done

rebuild: clean html
