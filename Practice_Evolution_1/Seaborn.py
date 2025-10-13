import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import load_dataset

# Load example dataset
tips = load_dataset("tips")

# ECDF for tip values
# sns.ecdfplot(data=tips, x="tip")

# plt.xlabel("Tip")
# plt.ylabel("ECDF")
# plt.title("ECDF of Tips")
# plt.show()


sns.kdeplot(data=tips, x="tip", fill=True)
plt.show()