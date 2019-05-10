from pymain import run_cnn
# l2 norm Teacher
b4, time = run_cnn('../../Net-Gen/TeacherNetworks/Teacher.h5', 200, '2')

# l2 norm Teacher
print("--------------------------------------------------")
print("------------------L2 Norm Teacher-----------------")
print("--------------------------------------------------")
print("Bound: " + str(b4))
