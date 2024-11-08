# Shopify (shopify/)

Scripts for working with an existing Shopify instance for the purpose of training, testing and serving ML recommendation models.

## Inventory management

The following scripts are part of the configuration:
* `inventory.py` specifies the desired state of the inventory in the existing Shopify instance
* `config.py` specifies the Shopify instance configuration parameters

Run the following scripts:
* `populate_inventory.py` to update the inventory of the Shopify instance to the state described in the `inventory.py`
* `cleanup.py` to cleanup the state of the inventory of the Shopify instance

Make sure the user of the token specified in `config.py`, has the appropriate `write_*` and `read_*` permissions for
manipulating collections, products and publications.

# ML recommendation model (ml/)

Steps:

1. Generate the interactions dataset that will be used for model training:
```shell
$ python generate_interactions.py
```
Note that currently this script requires access to the Shopify instance to retrieve the product handles mappings. Therefore
the credentials specified in `shopify/config.py` have to be valid.

Modify the interaction rules defined in `intraction_rules.py` to adjust the patterns between products that the model
should try to capture.

2. Train the model on the generated interactions dataset:
```shell
$ python train.py <interactions_dataset_file>
```
Each invocation of the training script will create a dedicated directory for the training session purposes. The trained
model, training savepoints & metrics, model metadata and vocabularies (required for model deployment) will reside in the
appropriate session directory.

3. Deploy the model to the Databricks MLFlow:
```shell
$ python deploy_databricks.py <path to model training session directory>
```
