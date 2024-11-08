SHOPIFY_STORE = "<INSERT STORE NAME HERE>"
ACCESS_TOKEN = "<INSERT ACCESS TOKEN HERE"
GRAPHQL_URL = f"https://{SHOPIFY_STORE}/admin/api/2023-10/graphql.json"

SHOPIFY_HEADERS = {
    "Content-Type": "application/json",
    "X-Shopify-Access-Token": ACCESS_TOKEN
}
