# Makefile for the installation
.ONESHELL:
SHELL:=/bin/bash

# To be defined
ENV_NAME:=crossmae
PYTHON_VERSION:=3.10
BASE:=~/miniconda3/envs/$(ENV_NAME)# NOTE: May need to modify this
BIN:=$(BASE)/bin

# Commands
CREATE_COMMAND:="conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y"
# CREATE_COMMAND="conda env create -f env.yml python=$(PYTHON_VERSION) --force -q"
DELETE_COMMAND:="conda env remove -n $(ENV_NAME)"
ACTIVATE_COMMAND:="conda activate -n $(ENV_NAME)"
DEACTIVATE_COMMAND:="conda deactivate"

# Lambdas
# Remove suffix "@ file" from requirements.txt
# remove_suffix = $(patsubst %@%,%,$1)# Not used

# To load a env file use env_file=<path to env file>
# e.g. make release env_file=.env
ifneq ($(env_file),)
	include $(env_file)
#	export
endif

all:
	$(MAKE) help
help:
	@echo
	@echo "+-------------------------------------------------------------------------------+"
	@echo "+                                DISPLAYING HELP                                +"
	@echo "+-------------------------------------------------------------------------------+"
	@echo
	@echo "make help"
	@echo "       Display this message"
	@echo "make install [env_file=<(optional) path to .env file>]"
	@echo "       Call delete_conda_env create_conda_env"
	@echo "make delete_env [env_file=<(optional) path to .env file>]"
	@echo "       Delete the current conda env or virtualenv"
	@echo "make create_env [env_file=<(optional) path to .env file>]"
	@echo "       Create a new conda env or virtualenv for the specified python version"
	@echo
	@echo "---------------------------------------------------------------------------------"
install:
	$(MAKE) delete_env
	$(MAKE) create_env
	@echo -e "\033[0;31m############################################"
	@echo
	@echo "Installation Successful!"
	@echo "To activate the conda environment run:"
	@echo '    conda activate sat_env'
clean:
	$(BIN)/python setup.py $(ENV_NAME)
delete_env:
	@echo "Deleting virtual environment.."
	eval $(DELETE_COMMAND)
create_env:
	@echo "Creating virtual environment.."
	eval $(CREATE_COMMAND)
# requirements:
# 	# awk '{print gensub(/ @.*/,"",1,$$0)}' requirements.txt > requirements.txt
# 	$(BIN)/pip install -r requirements.txt
# 	# conda env update --file env.yml --prune -n $(ENV_NAME)

# .PHONY: help install delete_env create_env requirements
.PHONY: help install delete_env create_env

