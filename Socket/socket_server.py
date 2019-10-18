import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('192.168.1.13', 8080))
s.listen(5)

clientsocket, address = s.accept()
print(f"Connection from {address} has been established.")
while True:
    # now our endpoint knows about the OTHER endpoint.
    motorA = 100
    motorB = 0
    x = str(motorA)+" "+str(motorB)
    time.sleep(0.01)
    clientsocket.send(bytes(x,"utf-8"))



