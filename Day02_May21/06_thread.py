import threading
import time


def task1():
    print(f"Thread task1 starting.")
    time.sleep(2)
    for i in range(5):
        print("Task 1 -", i)
    print(f"Thread task1 done.")


def task2():
    print(f"Thread task2 starting.")
    time.sleep(2)
    for i in range(5):
        print("Task 2 -", i)
    print(f"Thread task2 done.")


# 创建线程
thread1 = threading.Thread(target=task1)
thread2 = threading.Thread(target=task2)

# 启动线程
thread1.start()
thread2.start()

# 确保主线程等待子线程完成
thread1.join()
thread2.join()
print("All tasks are completed.")
