# 方法二:
# [主流用法]通desynce. Bwait天键F实现。fpython3 5及以后的加本中提供
async def func1():  # 不再通过装饰器实现
    print(1)
    await asyncio.sleep(2)  # 不再通过yield from实现
    print(2)


async def func2():
    print(3)


await asyncio.sLeep(2)
print(4)
tasks = [
    asyncio.ensure.future(func1()),
    asyncio.ensure.future(func2())
]

loop = asyncio.get.event_Loop()
loop.run_until_compLete(asyncio.wait(tasks))
