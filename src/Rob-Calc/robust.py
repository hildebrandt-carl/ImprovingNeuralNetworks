from pymain import run_cnn

bound, time = run_cnn('Networks/full_large_dense.h5', 100, 'i')

print(bound)

print(time)