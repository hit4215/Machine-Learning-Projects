import pandas as pd
import matplotlib.pyplot as plt

dataframe = {
    'Name':['Hit','Meet','Keet','Leet'],
    'Marks':[30,40,80,40]
}

df = pd.DataFrame(dataframe)

plt.bar(df["Name"],df["Marks"])
plt.xlabel("Student")
plt.ylabel("Marks")
plt.title("Bar Chart of Student Marks")
plt.show()