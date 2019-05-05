from pymain import run_cnn
# l-infinity norm
#bound, time = run_cnn('../Net-Gen/TeacherNetworks/NoTeacher.h5', 1000, 'i')
# l-infinity norm
#bound, time = run_cnn('../Net-Gen/TeacherNetworks/Teacher.h5', 1000, 'i')

# l1 norm
bound, time = run_cnn('../Net-Gen/TeacherNetworks/NoTeacher.h5', 1000, '1')
# l1 norm
#bound, time = run_cnn('../Net-Gen/TeacherNetworks/Teacher.h5', 1000, '1')