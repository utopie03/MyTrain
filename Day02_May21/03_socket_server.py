import socket


def socket_server():
    # 1.创建TCP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2.绑定地址和端口
    server_address = ("localhost", 12345)
    server_socket.bind(server_address)
    # 3.监听连接。一般最多5个连楼
    server_socket.listen(5)
    while True:
        print("waiting for a connection...")
        try:
            # 4.接收客户端连接
            client_socket, client_address = server_socket.accept()
            print(f"接收了客户端的连接，客户端的信息为: {client_address}")

            while True:
                # 5.接收数据
                data = client_socket.recv(1824)
                if not data:
                    print(client_address + "断开连接")
                    break
                else:
                    print(f"接收的数据: {data.decode('utf-8')}")

                # 6.发送数据
                message = f"hello, {data.decode('utf-8')}"
                client_socket.send(message.encode("utf-8"))

        except ConnectionResetError:
            print(client_address + "异常断开连接")
            continue

        except KeyboardInterrupt:
            print('服务器关闭')
            break

    # 7.关闭连楼
    server_socket.close()


if __name__ == '__main__':
    socket_server()
