import os
import re
import numpy as np
import matplotlib.pyplot as plt
directory = '../Net-Gen/FinalNetworks/'

filenames = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filenames.append(filename)


network_num = []
network_acc = []
network_ad_acc = []
network_N_acc = []

filenames.sort()
for filenum in range(0, len(filenames)):
    print("Processing: " + str(filenames[filenum]))
    with open(directory + filenames[filenum]) as f:
        for line in f:
            if "accuracy" in line:
                acc = re.findall("\d+\.\d+", line)
                acc = acc[0]

                network_num.append(filenum)
                network_acc.append(float(acc))


print("Minimum accuracy:" + str(min(network_acc)))
print("Maximum accuracy: " + str(max(network_acc)))

plt.bar(network_num, network_acc)
plt.xlabel("Network Number")
plt.ylabel("Accuracy (%)")
plt.title("Testing Accuracy of Individual Networks")
plt.ylim((0.75, 1)) 
plt.show()


# Getting the adversarial attacks
Nversion = False
with open("../Net-Use/Results/compare_results.txt") as f:
    for line in f:
        if Nversion == False:
            if "Accuracy" in line:
                acc = re.findall("\d+\.\d+", line)
                acc = acc[0]

                network_ad_acc.append(float(acc))
        
            if "N-Version Programming" in line:
                Nversion = True
        else:
            if "Accuracy" in line:
                acc = re.findall("\d+\.\d+", line)
                acc = acc[0]

                network_N_acc.append(float(acc))


p1 = plt.bar(network_num, network_acc)
p2 = plt.bar(network_num, network_ad_acc)

plt.ylabel('Accuracy (%)')
plt.title('Testing Accuracy vs Adversarial Dataset Accuracy')
plt.legend((p1[0], p2[0]), ('Testing Dataset', 'Adversarial Dataset'),loc='lower right')
plt.ylim((0.75, 1)) 
plt.show()

print("Average : " + str(np.mean(network_ad_acc)))
print("Max : " + str(max(network_ad_acc)))



plt.plot(network_num, network_N_acc)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel('Accuracy (%)')
plt.title('N-Version Programming Accuracy on Adversarial Data as N Increases')
plt.ylim((0.75, 1)) 
plt.show()

print("Best : " + str(max(network_N_acc)))


p1 = plt.plot(network_num, network_N_acc)
p2 = plt.bar(network_num, network_ad_acc, color='orange')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel('Accuracy (%)')
plt.title('N-Version Programming Accuracy on Adversarial Data as N Increases')
plt.legend((p1[0], p2[0]), ('N-Version Program', 'Individual Network'),loc='lower right')
plt.ylim((0.75, 1)) 
plt.show()


# TMR Plot

p1 = plt.scatter(network_ad_acc, network_N_acc)
plt.show()
# X- Component
# Y - System


