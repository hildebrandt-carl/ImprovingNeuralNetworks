from pymain import run_cnn
# l1 norm NoTeacher
b1, time = run_cnn('../../Net-Gen/TeacherNetworks/NoTeacher.h5', 5, '1')


# l1 norm NoTeacher
print("--------------------------------------------------")
print("----------------L1 Norm No Teacher----------------")
print("--------------------------------------------------")
print("Bound: " + str(b1))
