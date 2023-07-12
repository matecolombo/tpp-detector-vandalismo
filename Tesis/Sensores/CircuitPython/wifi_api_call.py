import os, time, ssl, wifi, socketpool, adafruit_requests
import board

# Log into Wifi network ussing credentials in settings.tonl
wifi.radio.connect(os.getenv("WIFI_SSID"), os.getenv("WIFI_PASSWORD"))
print("Connected to Wifi")

# sockets set up an endpoint for communication as a reusable 'pool'
pool = socketpool.SocketPool(wifi.radio)
# create an object so that a request for web data can be made
requests = adafruit_requests.Session(pool, ssl.create_default_context())

# to get time, setup the correct url for api call
url = "https://worldtimeapi.org/api/timezone/"
timezone = "America/Argentina/Buenos_Aires"
url = url + timezone

print(f"Accessing url:\n{url}")
response = requests.get(url)
print(f"The Api returned the text:\n{response.text}")