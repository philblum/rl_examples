import matplotlib.pyplot as plt

with 
plt.plot(np.mean(all_reward_sums[algorithm], axis=0), label=algorithm)
plt.xlabel("Episodes")
plt.ylabel("Sum of\n rewards\n during\n episode", rotation=0, labelpad=50)
plt.xlim(0, 250)
plt.ylim(-500, 1000)
plt.legend()
plt.grid(True)
plt.savefig('fig1.pdf')
plt.show()
