CONDA_ROOT = $(shell conda info --base)
CONDA_ENV_NAME = mnist-pytorch
PYTHON = $(CONDA_ROOT)/envs/$(CONDA_ENV_NAME)/bin/python3

SRC_DIR = src
DATA_DIR = data

TRAIN_DIR = $(DATA_DIR)/train
TEST_DIR = $(DATA_DIR)/test
SET_DIRS = $(TRAIN_DIR) $(TEST_DIR)

DOWNLOAD_URL_PREFIX = http://yann.lecun.com/exdb/mnist
IDX_IMAGES_SUFFIX = images-idx3-ubyte
IDX_LABELS_SUFFIX = labels-idx1-ubyte

TRAINING_DATA = $(TRAIN_DIR)/train-$(IDX_IMAGES_SUFFIX) \
	$(TRAIN_DIR)/train-$(IDX_LABELS_SUFFIX)

MODEL_SAVE = model.tar
.PRECIOUS: $(MODEL_SAVE)

# All the directories that need to be created.
MK_DIRECTORIES = $(SET_DIRS)

.PHONY: all
all: $(MODEL_SAVE)

# Setup the conda environment with all the required packages.
.PHONY: environment
environment:
	conda env create -f environment.yml -n $(CONDA_ENV_NAME)


# Download zipped data
.PRECIOUS: $(TRAIN_DIR)/%.gz $(TEST_DIR)/%.gz
$(TRAIN_DIR)/%.gz: | $(TRAIN_DIR)
	wget $(DOWNLOAD_URL_PREFIX)/$(@F) -O $@
$(TEST_DIR)/%.gz: | $(TEST_DIR)
	wget $(DOWNLOAD_URL_PREFIX)/$(@F) -O $@
# Unzip downloaded data
$(TRAIN_DIR)/%: $(TRAIN_DIR)/%.gz
	gzip -cdk $< > $@
$(TEST_DIR)/%: $(TEST_DIR)/%.gz
	gzip -cdk $< > $@
# There is surely a way to more concisely define these rules.
# Maybe something like this ? (SET=train or SET=t10k)
# ($(DATA_DIR)/$(SET)/$(SET)-%.gz: | $(DATA_DIR)/$(SET)
# 	  wget $(DOWNLOAD_URL_PREFIX)/$(@F) -O $@

# "Dependencies" of source code files.
#$(SRC_DIR)/CNN.py: $(SRC_DIR)/pytorch_model.py
#	@touch $@
#$(SRC_DIR)/train.py: $(SRC_DIR)/data.py $(SRC_DIR)/CNN.py
#	@touch $@

# Train and save model.
MODEL_SCRIPT = $(SRC_DIR)/CNN.py
TRAINING_SCRIPT = $(SRC_DIR)/train.py

TRAINING_SCRIPT_DEPS = $(SRC_DIR)/pytorch_model.py $(MODEL_SCRIPT)
$(MODEL_SAVE): $(TRAINING_SCRIPT) $(TRAINING_SCRIPT_DEPS) \
		$(TRAINING_DATA)
	$(PYTHON) $(TRAINING_SCRIPT) $(TRAINING_DATA) $@
#$(MODEL_SAVE): $(SRC_DIR)/train.py $(TRAINING_DATA)
#	$(PYTHON) $^ $@	

$(MK_DIRECTORIES):
	@mkdir -p $@
