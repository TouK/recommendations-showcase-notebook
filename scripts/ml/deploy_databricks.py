import sys
import os
import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from typing import Optional, Dict, Any
from mlflow.types import Schema, ColSpec, TensorSpec
from mlflow.models import ModelSignature

class SliRecModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
        from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
        from recommenders.models.deeprec.deeprec_utils import prepare_hparams

        hparams = prepare_hparams(
            context.artifacts["model_config"],
            user_vocab=context.artifacts["user_vocab"],
            item_vocab=context.artifacts["item_vocab"],
            cate_vocab=context.artifacts["category_vocab"],
        )

        self.model = SeqModel(hparams, SequentialIterator)
        self.model.load_model(context.artifacts["model_data"] + "/best_model")
        self.item_cat_dict = self.load_dict(context.artifacts["item_category_dict"])
        self.TOP_N_HIGHEST_RECOMMENDATIONS = 12
        self.BATCH_SIZE = 16

    def predict(self, context, model_input):
        import numpy as np

        user_id = model_input['userId'][0]
        history_item_ids = model_input['itemIds'][0]
        history_timestamps = model_input['timestamps'][0]

        batches = self.build_batches_generator(user_id, history_item_ids, history_timestamps)

        # TODO: confirm the type conversions and collection operations are optimal here
        inferenced_items = []
        inferenced_preds = []
        for (batch, items) in batches:
            preds = self.model.infer(self.model.sess, batch)
            inferenced_preds += (preds[0].flatten().tolist())
            inferenced_items += items

        top_n_item_indices = np.argsort(inferenced_preds)[::-1][:self.TOP_N_HIGHEST_RECOMMENDATIONS]
        return [{
            "items": np.array(inferenced_items)[top_n_item_indices].tolist(),
            "preds": np.array(inferenced_preds)[top_n_item_indices].tolist()
        }]

    def build_batches_generator(self, user_id, history_item_ids, history_timestamps, batch_num_ngs=0, min_seq_length=1):
        import time

        now_timestamp = int(time.time())
        it = self.model.iterator
        history_item_categories = [self.item_cat_dict[k] for k in history_item_ids]

        batched_item_ids = []
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []

        cnt = 0
        for item_id, item_category in self.item_cat_dict.items():
            encoded_item = self.encode_single_item(
                user_id,
                item_id,
                item_category,
                now_timestamp,
                history_item_ids,
                history_item_categories,
                history_timestamps
            )

            if len(encoded_item["historyItemIds"]) < min_seq_length:
                continue

            batched_item_ids.append(item_id)
            user_list.append(encoded_item["userId"])
            item_list.append(encoded_item["itemId"])
            item_cate_list.append(encoded_item["itemCategory"])
            item_history_batch.append(encoded_item["historyItemIds"])
            item_cate_history_batch.append(encoded_item["historyCategories"])
            time_list.append(encoded_item["currentTime"])
            time_diff_list.append(encoded_item["timeDiff"])
            time_from_first_action_list.append(encoded_item["timeFromFirstAction"])
            time_to_now_list.append(encoded_item["timeToNow"])

            # label is useless for prediction but required for SliRec conversion utilities
            label_list.append(0)

            cnt += 1
            if cnt == self.BATCH_SIZE:
                res = it._convert_data(
                    label_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    item_history_batch,
                    item_cate_history_batch,
                    time_list,
                    time_diff_list,
                    time_from_first_action_list,
                    time_to_now_list,
                    batch_num_ngs,
                )
                batch_feed_dict = it.gen_feed_dict(res)
                yield (batch_feed_dict, batched_item_ids) if batch_feed_dict else None

                batched_item_ids = []
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                cnt = 0
        # process the remaining inputs in the last batch
        if cnt > 0:
            res = it._convert_data(
                label_list,
                user_list,
                item_list,
                item_cate_list,
                item_history_batch,
                item_cate_history_batch,
                time_list,
                time_diff_list,
                time_from_first_action_list,
                time_to_now_list,
                batch_num_ngs,
            )
            batch_feed_dict = it.gen_feed_dict(res)
            yield (batch_feed_dict, batched_item_ids) if batch_feed_dict else None

    def encode_single_item(self, userId, itemId, itemCategory, nowTimestamp, historyItemIds, historyCategories, historyTimestamps):
        import numpy as np
        it = self.model.iterator

        user_id = it.userdict[userId] if userId in it.userdict else 0
        item_id = it.itemdict[itemId] if itemId in it.itemdict else 0
        item_cate = it.catedict[itemCategory] if itemCategory in it.catedict else 0
        current_time = float(nowTimestamp)

        item_history_sequence = []
        cate_history_sequence = []
        time_history_sequence = []

        for item in historyItemIds:
            item_history_sequence.append(
                it.itemdict[item] if item in it.itemdict else 0
            )

        for cate in historyCategories:
            cate_history_sequence.append(
                it.catedict[cate] if cate in it.catedict else 0
            )

        time_history_sequence = [float(i) for i in historyTimestamps]
        time_range = 3600 * 24

        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (time_history_sequence[i + 1] - time_history_sequence[i]) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)

        last_diff = (current_time - time_history_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.log(time_diff)

        time_from_first_action = []
        first_time = time_history_sequence[0]
        time_from_first_action = [
            (t - first_time) / time_range for t in time_history_sequence[1:]
        ]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (current_time - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = []
        time_to_now = [(current_time - t) / time_range for t in time_history_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
        time_to_now = np.log(time_to_now)

        return {
            "userId": user_id,
            "itemId": item_id,
            "itemCategory": item_cate,
            "historyItemIds": item_history_sequence,
            "historyCategories": cate_history_sequence,
            "currentTime": current_time,
            "timeDiff": time_diff,
            "timeFromFirstAction": time_from_first_action,
            "timeToNow": time_to_now
        }

    def load_dict(self, filename):
        import pickle as pkl

        with open(filename, "rb") as f:
            return pkl.load(f)

def usage():
    print(f"Usage: {sys.argv[0]} <model path>")

if len(sys.argv) == 1:
    usage()
    sys.exit(0)

# this is the experiment name in the Databricks specific format
EXPERIMENT_NAME = "/Users/<INSERT_USER_NAME>/sli_rec_experiment"
DATABRICKS_CLIENT_ID = "<INSERT_CLIENT_ID>"
DATABRICKS_CLIENT_SECRET = "<INSERT_CLIENT_SECRET>"
DATABRICKS_HOST = "https://<INSERT_INSTANCE_ID>.azuredatabricks.net"

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_CLIENT_ID"] = DATABRICKS_CLIENT_ID
os.environ["DATABRICKS_CLIENT_SECRET"] = DATABRICKS_CLIENT_SECRET

mlflow.set_registry_uri("databricks")
mlflow.set_tracking_uri("databricks")

print(f"Log model to MLFlow registry at: {mlflow.get_tracking_uri()}")
mlflow.set_experiment(EXPERIMENT_NAME)

signature = infer_signature(model_input = {
    "userId": "A2R27T4HADWFFJ",
    "itemIds": ["A", "B", "C"],
    "timestamps": ["A", "B", "C"]
}, model_output = {
    "items": ["A", "B"],
    "preds": [2.14, 5.43]
})

default_conda_env = mlflow.pyfunc.get_default_conda_env()
default_conda_env['dependencies'].append('tensorflow=1.12.0')
default_conda_env['dependencies'].append('recommenders=0.2.0')

model_path = sys.argv[1]
model_dir = os.path.join(os.getcwd(), model_path)

artifacts = {
    "model_config": "model_serve_config.yaml",
    "model_data": os.path.join(model_dir, 'training/model'),
    "user_vocab": os.path.join(model_dir, 'user_vocab.pkl'),
    "item_vocab": os.path.join(model_dir, 'item_vocab.pkl'),
    "category_vocab": os.path.join(model_dir, 'category_vocab.pkl'),
    "item_category_dict": os.path.join(model_dir, 'item_cat_dict.pkl'),
}

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="sli_rec_shopify_model",
        python_model=SliRecModelWrapper(),
        conda_env=default_conda_env,
        artifacts=artifacts,
        registered_model_name="sli_rec_shopify_model",
        signature=signature
    )
