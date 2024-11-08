import random
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

import interaction_rules

sys.path.append(os.path.abspath("../shopify"))
import products
import inventory

# builds lookup index: product name -> List[interaction id]
def validate_and_build_interactions_index(interactions, product_handle_mappings):
    missing_products = []

    index = {}
    for idx, product_group in enumerate(interactions):
        for p in product_group:
            if p not in product_handle_mappings:
                missing_products.append(p)
            if p not in index:
                index[p] = []
            index[p].append(idx)

    return (missing_products, index)

# builds product name -> product handle mappings based on the current Shopify instance inventory
all_products = products.get_all_products()
products_handle_dict = {}
for p in all_products:
    products_handle_dict[p["title"]] = p["handle"]

(missing_products, freq_bought_together_within_cat_idx) = validate_and_build_interactions_index(
    interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_WITHIN_CATEGORY,
    products_handle_dict
)

if len(missing_products) > 0:
    print(f"The following products do not exist in Shopify instance: {missing_products}")
    sys.exit(0)

(missing_products, freq_bought_together_across_cat_idx) = validate_and_build_interactions_index(
    interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_ACROSS_CATEGORIES,
    products_handle_dict
)

if len(missing_products) > 0:
    print(f"The following products do not exist in Shopify instance: {missing_products}")
    sys.exit(0)


interactions = []
time_diff = timedelta(days=30)
users = [f"user_{i}" for i in range(1, interaction_rules.MAX_USERS)]
categories = list(inventory.PRODUCTS.keys())

# TODO: Possible improvements:
#   * restrict interactions with the same product
#   * randomize starting timestamp to get more interactions spread over time
#   * add a slight chance to start generating interactions with the product selected from the interaction rules (nested for loop),
#     instead of a product selected for a given interaction (outer for loop)
for user in users:
    num_interactions = random.randint(interaction_rules.MIN_NUM_INTERACTIONS_PER_USER, interaction_rules.MAX_NUM_INTERACTIONS_PER_USER)

    for _ in range(num_interactions):
        # pick a random category
        category = random.choice(categories)

        # pick a random product in the selected category
        start_date = datetime.now() - timedelta(days=30)
        product = random.choice(inventory.PRODUCTS[category])
        timestamp = start_date + timedelta(seconds=random.randint(0, int(time_diff.total_seconds())))
        interactions.append(['1', user, products_handle_dict[product], int(timestamp.timestamp()), category])

        num_interactions_with_the_product = random.randint(1, interaction_rules.MAX_NUM_INTERACTIONS_WITH_THE_PRODUCT)
        for _ in range(num_interactions_with_the_product):
            roll = random.random()

            if roll < interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_IN_CATEGORY_PROB and product in freq_bought_together_within_cat_idx:
                # create an interaction with a product frequently bought together in the same category
                next_product_group_idx = random.choice(freq_bought_together_within_cat_idx[product])
                next_product = random.choice(interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_WITHIN_CATEGORY[next_product_group_idx])
                timestamp = start_date + timedelta(seconds=random.randint(0, int(time_diff.total_seconds())))
                interactions.append(['1', user, products_handle_dict[next_product], int(timestamp.timestamp()), category])
            elif roll < (interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_ACROSS_CATEGORIES_PROB + interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_IN_CATEGORY_PROB) and product in freq_bought_together_across_cat_idx:
                # create an interaction with a product frequently bought together in the different category
                next_product_group_idx = random.choice(freq_bought_together_across_cat_idx[product])
                next_product = random.choice(interaction_rules.FREQUENTLY_BOUGHT_TOGETHER_ACROSS_CATEGORIES[next_product_group_idx])
                timestamp = start_date + timedelta(seconds=random.randint(0, int(time_diff.total_seconds())))
                interactions.append(['1', user, products_handle_dict[next_product], int(timestamp.timestamp()), category])
            else:
                # create an interaction with a random product (should have the lowest chance of happening)
                next_product = random.choice(inventory.PRODUCTS[category])
                timestamp = start_date + timedelta(seconds=random.randint(0, int(time_diff.total_seconds())))
                interactions.append(['1', user, products_handle_dict[next_product], int(timestamp.timestamp()), category])

            # NOTE: here might be a random chance to switch the `product` with the `next_product`

print(f"Total interactions generated: {len(interactions)}")

interactions_df = pd.DataFrame(interactions, columns=['label', 'user', 'product', 'timestamp', 'category'])

interactions_filepath = "interactions_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
interactions_df.to_csv(interactions_filepath, index=False, sep='\t', header=False)

print("Interactions dataset saved in:", interactions_filepath)
