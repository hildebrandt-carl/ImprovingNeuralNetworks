from pymain import run_cnn
# l1 norm Teacher
b2, time = run_cnn('../../Net-Gen/TeacherNetworks/Teacher.h5', 200, '1')

# l1 norm Teacher
print("--------------------------------------------------")
print("------------------L1 Norm Teacher-----------------")
print("--------------------------------------------------")
print("Bound: " + str(b2))
