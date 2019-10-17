import multiprocessing
from threading import *
import time

max_proc_sem = multiprocessing.Semaphore(4)
proc = []

def test():
    time.sleep(1)
    max_proc_sem.release()
    print("-----------RELEASED PROC--------------")
    return

for i in range(10):
    print('asking for',i)
    while(not max_proc_sem.acquire(False)):
        time.sleep(.5)
        print("waiting...")
    print("-----------ACQUIRED PROC--------------")
    proc.append(multiprocessing.Process(target=test))
    proc[-1].start()
    
for i in proc:
    i.join()