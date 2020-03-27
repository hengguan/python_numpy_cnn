import socket 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(("172.16.20.43", 12345))
s.listen(3)

while True:
    conn, addr = s.accept()

    data = conn.recv(1024).decode('utf-8')
    print(data)
    conn.sendall(data.encode('utf-8'))
    conn.close()
