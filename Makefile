SHELL=/bin/bash
PROJECT_NAME=tianshou
PROJECT_PATH=${PROJECT_NAME}/
LINT_PATHS=${PROJECT_PATH} test/ docs/conf.py examples/ setup.py

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

pytest:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	$(call check_install, pytest_xdist)
	pytest test --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing

mypy:
	$(call check_install, mypy)
	mypy ${PROJECT_NAME}

lint:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)
	flake8 ${LINT_PATHS} --count --show-source --statistics

format:
	# sort imports
	$(call check_install, isort)
	isort ${LINT_PATHS}
	# reformat using yapf
	$(call check_install, yapf)
	yapf -ir ${LINT_PATHS}

check-codestyle:
	$(call check_install, isort)
	$(call check_install, yapf)
	isort --check ${LINT_PATHS} && yapf -r -d ${LINT_PATHS}

check-docstyle:
	$(call check_install, pydocstyle)
	$(call check_install, doc8)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	pydocstyle ${PROJECT_PATH} && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	cd docs && make html && cd _build/html && python3 -m http.server

spelling:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)
	cd docs && make spelling SPHINXOPTS="-W"

clean:
	cd docs && make clean

commit-checks: format lint mypy check-docstyle spelling

.PHONY: clean spelling doc mypy lint format check-codestyle check-docstyle commit-checks
