import socket


def socket_client():
    # 1.创建TCP套接字
    client_socket = socket.socket(socket.AF_INET, socket.soCK_STREAM)

    # 2.连接服务器
    server_address = ("localhost", 12345)
    client_socket.connect(server_address)

    while True:
        # 3.发送数据
        message = input(">>").strip()
        if not message:
            continue
    client_socket.sendall(message.encode("utf-8"))

    # 接收数据
    data = client_socket.recv(1024)
    print(f"接收的数据:{data.decode("utf-8")}")


# 关团连接
client_socket.close()
