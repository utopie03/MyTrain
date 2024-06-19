import os
import time
from multiprocessing.pool import Pool


class Process(object):
    def __init__(self):
        print('init...')
        self.ab = 3

    def task(self, n):
        time.sleep(5)
        self.ab += n
        print("\n进程(%s)，收到%s，+n=%s" % (os.getpid(), n, self.ab))


if __name__ == '__main__':
    a = Process()
    pool_num = 4
    # 进程池
    p = Pool(pool_num)

    num_task = 12
    for i in range(num_task):
        # 进程池中添加任务
        t = p.apply_async(func=a.task, args=(i,))

    p.close()
    p.join()
    print("done: 主进程: aa.ab = %s" % a.ab)
