SHELL=/bin/bash
PROJECT_NAME=tianshou
PROJECT_PATH=${PROJECT_NAME}/
PYTHON_FILES = $(shell find setup.py ${PROJECT_NAME} test docs/conf.py examples -type f -name "*.py")

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

pytest:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	$(call check_install, pytest_xdist)
	pytest test --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing --color=yes

mypy:
	$(call check_install, mypy)
	mypy ${PROJECT_NAME}

lint:
	$(call check_install, ruff)
	$(call check_install, black)
	ruff check .
	black --check .

lint-format:
	$(call check_install, pre_commit)
	pre-commit run --all-files

doc:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install, sphinxcontrib.bibtex, sphinxcontrib_bibtex)
	cd docs && make html && cd _build/html && python3 -m http.server

spelling:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)
	$(call check_install, sphinxcontrib.bibtex, sphinxcontrib_bibtex)
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

clean: doc-clean

commit-checks: lint check-codestyle mypy check-docstyle spelling

.PHONY: clean spelling doc mypy lint format check-codestyle check-docstyle commit-checks
