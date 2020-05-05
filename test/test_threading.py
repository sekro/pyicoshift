import concurrent.futures
import math

PRIMES = [
    (112272535095293,'first'),
(112272535095293,'first'),
(112272535095293,'first'),
(112272535095293,'first'),
(112272535095293,'first'),
(112272535095293,'first'),
(112272535095293,'first'),
    (112582705942171,'second'),
    (112272535095293,'third'),
    (115280095190773,'fourth'),
    (115797848077099,'fifth'),
    (1099726899285419, 'sixth')]

mask = [4,5]

fix = 20

def is_prime(input, fix):
    n, txt = input
    k = 1
    print('inthread %i' % fix)
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False, txt, k
    return True, txt, k

def main():
    tasks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, input in enumerate(PRIMES):
            if i not in mask:
                tasks.append(executor.submit(is_prime, input, fix))
    for future in tasks:
        print(future.result())
    print(tasks)

if __name__ == '__main__':
    main()