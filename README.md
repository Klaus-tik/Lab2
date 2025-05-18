import numpy as np
import time
from scipy.linalg import blas

def generate_matrix(n):
    """Генерация случайной комплексной матрицы"""
    return np.random.rand(n, n) + 1j * np.random.rand(n, n)

def standard_multiplication(a, b):
    """Стандартное перемножение матриц по формуле из линейной алгебры"""
    n = a.shape[0]
    result = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            result[i, j] = np.sum(a[i, :] * b[:, j])
    return result

def blas_multiplication(a, b):
    """Перемножение матриц с использованием BLAS"""
    return blas.zgemm(alpha=1.0, a=a, b=b)

def blocked_multiplication(a, b, block_size=64):
    """Оптимизированное перемножение с разбиением на блоки"""
    n = a.shape[0]
    result = np.zeros((n, n), dtype=np.complex128)
    
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                i_end = min(ii + block_size, n)
                j_end = min(jj + block_size, n)
                k_end = min(kk + block_size, n)
                
                a_block = a[ii:i_end, kk:k_end]
                b_block = b[kk:k_end, jj:j_end]
                result[ii:i_end, jj:j_end] += np.dot(a_block, b_block)
    
    return result

def benchmark(multiplication_func, a, b, name):
    """Замер времени выполнения и расчет производительности"""
    start_time = time.perf_counter()
    result = multiplication_func(a, b)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    n = a.shape[0]
    complexity = 2 * n ** 3 
    mflops = complexity / elapsed_time * 1e-6
    
    print(f"{name}:")
    print(f"  Время выполнения: {elapsed_time:.2f} сек")
    print(f"  Теоретическая сложность: {complexity:.2e} операций")
    print(f"  Производительность: {mflops:.2f} MFlops")
    print()
    
    return result

def verify_results(res1, res2, name1, name2):
    """Проверка корректности результатов"""
    correlation = np.corrcoef(res1.ravel(), res2.ravel())[0, 1]
    print(f"Корреляция между {name1} и {name2}: {correlation:.6f}")
    print(f"Максимальное расхождение: {np.max(np.abs(res1 - res2)):.2e}")
    print()

if name == "__main__":
    np.random.seed(42)
    n = 2048
    
    print(f"Генерация матриц {n}x{n}...")
    a = generate_matrix(n)
    b = generate_matrix(n)
    print("Готово\n")
    
    test_n = 128
    print(f"Проверка корректности на матрицах {test_n}x{test_n}...")
    test_a = generate_matrix(test_n)
    test_b = generate_matrix(test_n)
    
    res_std = benchmark(standard_multiplication, test_a, test_b, "Тест: Стандартное")
    res_blas = benchmark(blas_multiplication, test_a, test_b, "Тест: BLAS")
    res_block = benchmark(blocked_multiplication, test_a, test_b, "Тест: Блочное")
    
    verify_results(res_std, res_blas, "Стандартное", "BLAS")
    verify_results(res_std, res_block, "Стандартное", "Блочное")
    
    print(f"\nОсновные замеры для матриц {n}x{n}:")
    
    # 1. BLAS - самый быстрый вариант
    res_blas = benchmark(blas_multiplication, a, b, "BLAS (zgemm)")
    
    # 2. Блочное перемножение
    res_block = benchmark(blocked_multiplication, a, b, "Блочное перемножение")
    verify_results(res_blas, res_block, "BLAS", "Блочное")
    
    # 3. Стандартное перемножение (очень медленно для больших матриц)
    if n <= 512:
        res_std = benchmark(standard_multiplication, a, b, "Стандартное перемножение")
        verify_results(res_blas, res_std, "BLAS", "Стандартное")
    else:
        print("Стандартное перемножение пропущено для n > 512 (слишком медленно)")
    
    print("\nАнализ производительности:")
    print("1. BLAS показывает наивысшую производительность, используя все оптимизации CPU")
    print("2. Блочный алгоритм должен достигать 30-70% от производительности BLAS")
    print("3. Стандартный алгоритм крайне неэффективен для больших матриц")
