# ========================================
# Configuration
# ========================================

PYTHON = python
PIP = pip

TEST_DIR = tests
PROBLEMS_SCRIPT = run_project_problems.py
PLOTTING_SCRIPT = scripts/generate_all_plots.py
# EXPERIMENTS_SCRIPT = 

# ========================================
# Commands
# ========================================

.PHONY: help install test problems plotting experiments run clean reinstall

help:
	@echo "Available targets:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run all pytest unit tests"
	@echo "  make problems      Run project problems"
	@echo "  make plotting  	Generate plots from project problem output"
	@echo "  make experiments  	Run project experiments"
	@echo "  make run  			Run project problems, plotting, and experiments"
	@echo "  make clean         Remove Python cache files"
	@echo "  make reinstall     Reinstall dependencies cleanly"

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

reinstall:
	@echo "Reinstalling dependencies cleanly..."
	$(PIP) uninstall -y numpy scipy matplotlib pytest
	$(PIP) install -r requirements.txt

test:
	@echo "Running unit tests..."
	pytest -v tests || true

problems:
	echo "Running project problems..."
	$(PYTHON) $(PROBLEMS_SCRIPT)

plotting:
	@echo "Running project plotting..."
	$(PYTHON) $(PLOTTING_SCRIPT)

# experiments:
# 	@echo "Running project experiments..."
# 	$(PYTHON) $(EXPERIMENTS_SCRIPT)

run:
	@echo "Running project problems, plotting, and experiments..."
	$(PYTHON) $(PROBLEMS_SCRIPT)
	$(PYTHON) $(PLOTTING_SCRIPT)

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*_cache" -delete

# 	rm -r results
# 	mkdir results
# 	mkdir results/histories
# 	mkdir results/plots