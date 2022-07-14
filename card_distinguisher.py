# Card Distinguisher - By Keval Shah

import sensor, image, time, tf, pyb
import network, usocket, sys
from credentials import SSID, KEY # Hidden credentials file containing the SSID and key of my local network

model_file = "ei-test-project-transfer-learning-tensorflow-lite-int8-quantized-model.lite" # TensorFlow Lite Model to read from

labels = ["pokemon_card", "regular_card"] # Labels to distinguish between

ledRed = pyb.LED(1) # Red LED colour
ledGreen = pyb.LED(2) # Green LED colour

HOST = '' # Use first available interface
PORT = 80 # Arbitrary non-privileged port

# Set up sensor
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((96, 96))
sensor.skip_frames(time = 2000)

# Set up clock
clock = time.clock()

# Init wlan module and connect to network
print("Trying to connect... (may take a while)...")
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, KEY, timeout=30000)

# We should have a valid IP now via DHCP
print(wlan.ifconfig())

# Create server socket
s = usocket.socket(usocket.AF_INET, usocket.SOCK_STREAM)

# Bind and listen
s.bind([HOST, PORT])
s.listen(5)

# Set server socket to blocking
s.setblocking(True)

def start_streaming(s):
    print(wlan.ifconfig())
    print ('Waiting for connections..')
    client, addr = s.accept()
    # set client socket timeout to 2s
    client.settimeout(2.0)
    print ('Connected to ' + addr[0] + ':' + str(addr[1]))

    # Read request from client
    data = client.recv(1024)
    # Should parse client request here

    # Send multipart header
    client.send("HTTP/1.1 200 OK\r\n" \
                "Server: OpenMV\r\n" \
                "Content-Type: multipart/x-mixed-replace;boundary=openmv\r\n" \
                "Cache-Control: no-cache\r\n" \
                "Pragma: no-cache\r\n\r\n")

    # FPS clock
    clock = time.clock()

    # Start streaming images
    # NOTE: Disable IDE preview to increase streaming FPS.
    frame = sensor.snapshot()
    cframe = frame.compressed(quality=35)
    header = "\r\n--openmv\r\n" \
             "Content-Type: image/jpeg\r\n"\
             "Content-Length:"+str(cframe.size())+"\r\n\r\n"
    client.send(header)
    client.send(cframe)

while(True):
    clock.tick() # Advances clock
    img = sensor.snapshot() # Gets image from sensor
    img.set(h_mirror = True) # Mirrors image
    objs = tf.classify(model_file, img)  # Classifies image based on model
    predictions = objs[0].output() # Provides predictions based on classifications

    max_val = max(predictions) # Finds the most probable label
    max_index = predictions.index(max_val) # Finds the index of the most probable label

    if max_index == 0: # If the index is the first label then turn red LED off and turn green LED on
        ledRed.off()
        ledGreen.on()
    else: # If the index is not the first label then turn red LED on and turn green LED off
        ledRed.on()
        ledGreen.off()

    img.draw_string( # Draws text onto image showing the classification and value
        0,
        0,
        labels[max_index] + "\n{:.2f}".format(round(max_val, 2)),
        mono_space = False,
        scale=1
    )

    print("-----")

    for i, label in enumerate(labels): # Prints classification and value to console for debugging purposes
       print(str(label) + ": " + str(predictions[i]))

    print("FPS:", clock.fps()) # Prints FPS for debugging purposes

    try:
        start_streaming(s)
    except OSError as e:
        print("socket error: ", e)
        #sys.print_exception(e)
    except Exception as e:
        print("unknown error: " + str(e))
