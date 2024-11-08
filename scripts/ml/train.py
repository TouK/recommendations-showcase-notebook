import os
import sys
import datetime
import shutil
import tensorflow.compat.v1 as tf
import pandas as pd
import random
import pickle as pkl
tf.get_logger().setLevel('ERROR')

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing, _create_vocab, _data_generating, _create_item2cate, _data_processing
from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator

def _generate_negative_samples_for_file(dataset_lines, f, neg_nums_count, all_items, item_cat_dict):
    for line in dataset_lines:
        # write out the positive sample immediately
        f.write(line)

        # <label=1> <user_id> <item_id> <category_id> <timestamp> ...
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < neg_nums_count:
            # find a random negative item that the user has not interacted with yet
            neg_item = random.choice(all_items)
            if neg_item == positive_item or neg_item in neg_items:
                continue

            count += 1
            neg_items.add(neg_item)

            # append a negative interaction with a selected item
            words[0] = "0"
            words[2] = neg_item
            words[3] = item_cat_dict[neg_item]
            f.write("\t".join(words) + "\n")

def negative_sampling_offline(
    instance_input_file, valid_file, test_file, item_cat_dict, valid_neg_nums=4, test_neg_nums=49
):
    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_input_file, sep="\t", names=columns)
    items_with_popular = list(ns_df["item_id"])

    with open(valid_file, "r") as f:
        valid_lines = f.readlines()

    with open(valid_file, "w") as f:
        _generate_negative_samples_for_file(valid_lines, f, valid_neg_nums, items_with_popular, item_cat_dict)

    with open(test_file, "r") as f:
        test_lines = f.readlines()

    with open(test_file, "w") as f:
        _generate_negative_samples_for_file(test_lines, f, test_neg_nums, items_with_popular, item_cat_dict)

# Data set preparation
if len(sys.argv) == 1:
    print("Provide the interactions dataset!!!")
    sys.exit(0)

dataset_csv = sys.argv[1]
print(f"[+] Using dataset from: {dataset_csv}")

TRAIN_DIR = "training"

training_dir = os.path.join(os.getcwd(), TRAIN_DIR)
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

session_dir = os.path.join(training_dir, "sess_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(session_dir):
    os.makedirs(session_dir)

print(f"[+] Training session location: {session_dir}")

# copy the training dataset to a session folder
instance_file = os.path.join(session_dir, dataset_csv)
shutil.copyfile(dataset_csv, instance_file)

# assign each interaction from the provided training dataset to the train/valid/test interactions
assigned_interactions = _data_processing(instance_file)

# expand the train/valid/test interactions history and persist it into file
valid_file = os.path.join(session_dir, r'valid_data')
test_file = os.path.join(session_dir, r'test_data')
train_file = os.path.join(session_dir, r'train_data')
_data_generating(assigned_interactions, train_file, valid_file, test_file)

# generate vocabularies for user/item/categories embeddings
user_vocab = os.path.join(session_dir, r'user_vocab.pkl')
item_vocab = os.path.join(session_dir, r'item_vocab.pkl')
cate_vocab = os.path.join(session_dir, r'category_vocab.pkl')
_create_vocab(train_file, user_vocab, item_vocab, cate_vocab)

# generate the item to category mappings
item_cat_dict = {}
with open(instance_file) as f:
    for line in f:
        arr = line.strip("\n").split("\t")
        mid = arr[2]
        cat = arr[4]
        item_cat_dict[mid] = cat

mapping_file = os.path.join(session_dir, r'item_cat_dict.pkl')
pkl.dump(item_cat_dict, open(mapping_file, "wb"))

# generate negative samples for validation and testing dataset
valid_num_ngs = 4 # number of negative instances with a positive instance for validation
test_num_ngs = 9 # number of negative instances with a positive instance for testing
negative_sampling_offline(instance_file, valid_file, test_file, item_cat_dict, valid_num_ngs, test_num_ngs)

# Begin model training

# Prepare model hyperparameters
EPOCHS = 10
BATCH_SIZE = 200
yaml_train_config_file = 'training/model_train_config.yaml'
train_num_ngs = 4 # number of negative instances with a positive instance for training
hparams = prepare_hparams(yaml_train_config_file,
                          embed_l2=0.,
                          layer_l2=0.,
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          show_step=30,
                          MODEL_DIR=os.path.join(session_dir, "training/model"),
                          SUMMARIES_DIR=os.path.join(session_dir, "training/summary/"),
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=True,
                          train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                          )

# Create model instance and begin training
model = SeqModel(hparams, SequentialIterator, seed=SEED)

with Timer() as train_time:
    # valid_num_ngs is the number of negative lines after each positive line in your valid_file
    # we will evaluate the performance of model on valid_file every epoch
    model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs)

print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))

# Evaluate model metrics on the test file
test_eval = model.run_eval(test_file, num_ngs=test_num_ngs)
print(test_eval)
