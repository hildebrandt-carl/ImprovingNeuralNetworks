from pymain import run_cnn
# l2 norm NoTeacher
b3, time = run_cnn('../../Net-Gen/TeacherNetworks/NoTeacher.h5', 5, '2')

# l2 norm NoTeacher
print("--------------------------------------------------")
print("----------------L2 Norm No Teacher----------------")
print("--------------------------------------------------")
print("Bound: " + str(b3))

