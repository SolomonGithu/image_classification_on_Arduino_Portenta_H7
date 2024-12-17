import sensor, image, time, os, ml, uos, gc
from ulab import numpy as np

import network
import socket
import time

# ***** set camera parameters *****#
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
#sensor.set_framesize(sensor.B320X320)  
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.
# ********************************************************#
net = None
labels = None

# ***** Wireless communication configuration *****#
# Wi-Fi credentials
SSID = ""           # Put your WiFi network SSID
PASSWORD = ""       # Put your WiFi password

# SMTP server details
SMTP_SERVER = "mail.smtp2go.com"
SMTP_PORT = 2525    # You can also try 25, 8025, or 587
# Base64 encoded username and password
SMTP_USER = ""      # Change this to your base64 encoded SMTP user's username
SMTP_PASSWORD  = "" # Change this to your base64 encoded SMTP user's password
# Note: In SMTP2GO, SMTP users are permitted to send emails over SMTP with a username and password

# Email details
FROM_EMAIL = ""     # Put the email address you used to Sign Up for SMTP2GO
TO_EMAIL = ""       # Put the receipient email
SUBJECT = "Urgent alert: Suspicious activity near your car!"
# The email message contains multiple lines.
MESSAGE_starting = "Hi there!\r\nA person has been spotted engaging in suspicious activity around your car's "
MESSAGE_threat = "" # contains the class: tyre or window
MESSAGE_ending = ". Don't wait—check your car now to ensure everything is secure.\r\n \r\nYour NVIDIA powered smart camera,\r\nArduino Portenta H7."

model_confidence_for_emailing = 0.8 # threshold to determine when an email will be sent based on the model's predictions
# ********************************************************#

# ***** Connect to Wi-Fi *****#
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

while not wlan.isconnected():
    print(f"Connecting to Wi-Fi SSID: {SSID}...")
    time.sleep(1)

print("Connected to Wi-Fi:", wlan.ifconfig())
# ********************************************************#

# ***** Function definition *****#
# Function to connect to SMTP server and send email
def send_email(threat_class):
    # Construct the message
    if threat_class == "potential_tyre_theft":
        MESSAGE_threat = "tyre"
    elif threat_class == "potential_window_theft":
        MESSAGE_threat = "window"
    try:
        addr = socket.getaddrinfo(SMTP_SERVER, SMTP_PORT)[0][4]
        print("SMTP server address:", addr)
        
        # Create a socket
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(addr)
        
        # Read server response
        def read_response():
            response = client.recv(1024).decode()
            print(response)
            return response
        
        # Send data
        def send_data(data):
            print(">>", data.strip())
            client.send((data + "\r\n").encode())
        
        # SMTP handshake and email sending
        read_response()  # Initial server response
        send_data(f"EHLO {wlan.ifconfig()[0]}")
        read_response()
        
        send_data("AUTH LOGIN")
        read_response()
        send_data(SMTP_USER)  # Base64 encoded username
        read_response()
        send_data(SMTP_PASSWORD)  # Base64 encoded password
        read_response()
        
        send_data(f"MAIL FROM:<{FROM_EMAIL}>")
        read_response()
        send_data(f"RCPT TO:<{TO_EMAIL}>")
        read_response()
        send_data("DATA")
        read_response()
        
        # Email content
        email_content = f"""\
To: {TO_EMAIL}
From: {FROM_EMAIL}
Subject: {SUBJECT}

{MESSAGE_starting + MESSAGE_threat + MESSAGE_ending}
."""
        send_data(email_content)
        read_response()
        
        send_data("QUIT")
        read_response()
        client.close()
        print("Email sent successfully!")
    except Exception as e:
        print("Failed to send email:", e)
# ********************************#

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    predictions_list = list(zip(labels, net.predict([img])[0].flatten().tolist()))

    """
    print("predictions_list[0][0]", predictions_list[0][0])
    print("predictions_list[1][0]", predictions_list[1][0])
    print("predictions_list[2][0]", predictions_list[2][0])
    # prints:
    #predictions_list[0][0] potential_tyre_theft
    #predictions_list[1][0] potential_window_theft
    #predictions_list[2][0] safe_car
    """

    for i in range(len(predictions_list)):
        print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))
    
    # TO DO: improve on the email triggers to prevent spamming someone's inbox :)
    # check if potential_tyre_theft confidence is > 0.8
    if(predictions_list[0][1] > 0.8):
        send_email("potential_tyre_theft")
    # check if potential_window_theft confidence is > 0.8
    elif(predictions_list[1][1] > 0.8):
        send_email("potential_window_theft")

    print(clock.fps(), "fps")

""" logs:
Connecting to Wi-Fi SSID: Lab...
Connecting to Wi-Fi SSID: Lab...
Connecting to Wi-Fi SSID: Lab...
Connecting to Wi-Fi SSID: Lab...
Connecting to Wi-Fi SSID: Lab...
Connecting to Wi-Fi SSID: Lab...
Connecting to Wi-Fi SSID: Lab...
Connected to Wi-Fi: ('XXX.XXX.XXX.XX', '255.255.255.0', 'XXX.XXX.XXX.X', 'XXX.XXX.XXX.X')
potential_tyre_theft = 0.125000
potential_window_theft = 0.386719
safe_car = 0.492188
0.379651 fps
SMTP server address: ('176.58.103.10', 2525)
220 mail.smtp2go.com ESMTP Exim 4.97.1-S2G Fri, 13 Dec 2024 20:59:01 +0000

>> EHLO 192.168.190.55
250-mail.smtp2go.com Hello XXX.XXX.XXX.XX [XXX.XX.XXX.XX]
250-SIZE 52428800
250-8BITMIME
250-DSN
250-PIPELINING
250-PIPECONNECT
250-AUTH CRAM-MD5 PLAIN LOGIN
250-CHUNKING
250-STARTTLS
250-PRDR
250 HELP

>> AUTH LOGIN
334 XXXXXXXXXXXXXXXXXXX

>> XXXXXXXXXXXXXXXXX
334 XXXXXXXXXXXXX

>> XXXXXXXXXXXXXXXXXXX
235 Authentication succeeded

>> MAIL FROM:<XXXXXXXXXXXXXXXXXXX>
250 sender ok.

>> RCPT TO:<XXXXXXXXXXXXXXXXXXXXX>
250 Accepted <XXXXXXXXXXXXXXXXXXXXXX>

>> DATA
354 Enter message, ending with "." on a line by itself

>> To: XXXXXXXXXXXXXXXXXXX
From: XXXXXXXXXXXXXXXXXXXXXXXXX
Subject: Urgent alert: Suspicious ativity near your car!

Hi there!
A person has been spotted engaging in suspicious activity around your car's window. Don't wait—check your car now to ensure everything is secure.
 
Your NVIDIA powered smart camera,
Arduino Portenta H7.
.
250 OK id=xxxxxxxxxxxxxxxxx

>> QUIT
221 mail.smtp2go.com closing connection

Email sent successfully!
"""