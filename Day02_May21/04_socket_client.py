import socket


def socket_client():
    # 1.����TCP�׽���
    client_socket = socket.socket(socket.AF_INET, socket.soCK_STREAM)

    # 2.���ӷ�����
    server_address = ("localhost", 12345)
    client_socket.connect(server_address)

    while True:
        # 3.��������
        message = input(">>").strip()
        if not message:
            continue
    client_socket.sendall(message.encode("utf-8"))

    # ��������
    data = client_socket.recv(1024)
    print(f"���յ�����:{data.decode("utf-8")}")


# ��������
client_socket.close()
