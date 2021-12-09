#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = dyme
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.7
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
endif

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip uninstall -y tensorboard
	$(PYTHON_INTERPRETER) -m spacy download en
	$(PYTHON_INTERPRETER) -m nltk.downloader stopwords

## Download Datasets
download_dailydialog_dataset:
### abort if dailydialog folder already exists; else create necessary folders, download, unzip and clean directory
	@if [ -d datasets/dailydialog ]; \
		then echo "Folder for DailyDialog dataset already exists! Please delete the folder 'dailydialog' if you want to re-download the dataset."; \
	else \
		if ! [ -d datasets/dailydialog ]; then \
			echo "Creating folder 'datasets/dailydialog'..." && \
			mkdir -p datasets/dailydialog && \
			echo "Folder 'datasets/dailydialog' successfully created." && \
			echo "Downloading dailydialog dataset..." && \
			curl http://yanran.li/files/ijcnlp_dailydialog.zip >> datasets/dailydialog/data.zip && \
			echo "Dailydialog dataset successfully downloaded." && \
			echo "Unpacking dailydialog dataset..." && \
			unzip -j -o datasets/dailydialog/data.zip -d datasets/dailydialog && \
			echo "Dailydialog dataset successfully unpacked." && \
			echo "Removing zip file..." && \
			rm datasets/dailydialog/data.zip && \
			echo "Dailydialog dataset setup successful!"; \
		fi \
	fi

download_empatheticdialogues_dataset:
### abort if dailydialog folder already exists
	@if [ -d datasets/empatheticdialogues ]; \
		then echo "Folder for EmpatheticDialogues dataset already exists! Please delete the folder 'empatheticdialogues' if you want to re-download the dataset."; \
	else \
		if ! [ -d datasets/empatheticdialogues ]; then \
			echo "Creating folder 'datasets/empatheticdialogues'..." && \
			mkdir -p datasets/empatheticdialogues && \
			echo "Folder 'datasets/empatheticdialogues' successfully created." && \
			echo "Downloading empatheticdialogues dataset..." && \
			curl https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz >> datasets/empatheticdialogues/data.tar.gz && \
			echo "EmpatheticDialogues dataset successfully downloaded." && \
			echo "Unpacking empatheticdialogues dataset..." && \
			tar -xvf datasets/empatheticdialogues/data.tar.gz -C datasets/empatheticdialogues --strip-components 1 && \
			echo "EmpatheticDialogues dataset successfully unpacked." && \
			echo "Removing zip file..." && \
			rm datasets/empatheticdialogues/data.tar.gz && \
			echo "EmpatheticDialogues dataset setup successful!"; \
		fi \
	fi