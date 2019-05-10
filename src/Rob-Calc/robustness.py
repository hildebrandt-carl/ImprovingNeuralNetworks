import sys
from pymain import run_cnn

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("Please input the measure and network name")
  exit()

network_name = sys.argv[1]
measure = sys.argv[2]

if network_name == "Teacher":
    if measure == "L1":
        # l1 norm Teacher
        b2, time = run_cnn('../Net-Gen/TeacherNetworks/Teacher.h5', 200, '1')
        # l1 norm Teacher
        print("--------------------------------------------------")
        print("------------------L1 Norm Teacher-----------------")
        print("--------------------------------------------------")
        print("Bound: " + str(b2))
    elif measure == "L2":
        # l2 norm Teacher
        b4, time = run_cnn('../Net-Gen/TeacherNetworks/Teacher.h5', 200, '2')
        # l2 norm Teacher
        print("--------------------------------------------------")
        print("------------------L2 Norm Teacher-----------------")
        print("--------------------------------------------------")
        print("Bound: " + str(b4))

elif network_name == "NoTeacher":
    if measure == "L1":
        # l1 norm NoTeacher
        b1, time = run_cnn('../Net-Gen/TeacherNetworks/NoTeacher.h5', 200, '1')
        # l1 norm NoTeacher
        print("--------------------------------------------------")
        print("----------------L1 Norm No Teacher----------------")
        print("--------------------------------------------------")
        print("Bound: " + str(b1))
    elif measure == "L2":
        # l2 norm NoTeacher
        b3, time = run_cnn('../Net-Gen/TeacherNetworks/NoTeacher.h5', 200, '2')
        # l2 norm NoTeacher
        print("--------------------------------------------------")
        print("----------------L2 Norm No Teacher----------------")
        print("--------------------------------------------------")
        print("Bound: " + str(b3))