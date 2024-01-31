import os 
import time
import threading

print(f"{os.cpu_count()} CPU cores on this machine")

def sleep_for(seconds):
    print(f"Sleeping for {seconds} second(s) ...")
    time.sleep(seconds)
    print("Done Sleeping ...")


threads = []
n_threads = int(input("\nHow many threads do you want to create ? \n"))

start = time.time()

for _ in range(n_threads):
    t = threading.Thread(target=sleep_for, args=(1.6,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

end = time.time()

print(f"Program ran in {round(end - start, 2)} second(s)")
