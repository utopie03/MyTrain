import socket

# 1．创建套接字对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2．绑定地址和端口
server_socket.bind(('localhost', 12545))

# 3. 开始监听连接5
server_socket.listen(1)
print('Wating for connection...')

# 4，接受连接
client_socket, client_addr = server_socket.accept()
print('connection from', client_addr)

# 5．接收数据
data = client_socket.recv(1024)
print('Received', data.decode())

# 6．发送响应数据
client_socket.send(b'Hello,client!')

# 7．关闭连接
client_socket.close()
server_socket.close()

#10.176.140.40