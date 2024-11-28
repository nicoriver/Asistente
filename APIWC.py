from woocommerce import API
# Reemplaza con tus credenciales
wc = API(
    url="https://desarrollosaltouruguay.com.ar/cosmeticosrosana/",
    consumer_key="ck_6dcb08d87a33708f8fd38075f8c91f7c3aece14e",
    consumer_secret="cs_44f69f355fdd41f780f7f8c2396de23cab7620e0",
    version="wc/v3"
)
# Obtener todos los productos
products = wc.get("products")
print(products)