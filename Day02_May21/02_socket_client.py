import socket

# 1．创建套接字对象
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2．连接服务器
client_socket.connect(('localhost', 12345))

# 3．发送数据
client_socket.send(b'Hello Server!')

# 4．接收响应数据
data = client_socket.recv(1024)
print('Received: ', data.decode())

# 5．关闭连接
client_socket.close()
