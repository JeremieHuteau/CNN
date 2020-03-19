SRC_DIR = src
DATA_DIR = data

TRAIN_DIR = $(DATA_DIR)/train
TEST_DIR = $(DATA_DIR)/test
SET_DIRS = $(TRAIN_DIR) $(TEST_DIR)

DOWNLOAD_URL_PREFIX = http://yann.lecun.com/exdb/mnist
IDX_IMAGES_SUFFIX = images-idx3-ubyte
IDX_LABELS_SUFFIX = labels-idx1-ubyte

# All the directories that need to be created.
MK_DIRECTORIES = $(SET_DIRS)

.PHONY: all
all: $(TRAIN_DIR)/train-$(IDX_LABELS_SUFFIX) \
	$(TRAIN_DIR)/train-$(IDX_IMAGES_SUFFIX)


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

$(MK_DIRECTORIES):
	mkdir -p $@
