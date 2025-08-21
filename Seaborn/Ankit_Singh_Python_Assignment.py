import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
df = pd.read_csv("sale_data.csv", nrows=20)
# print(df) 

plt.plot(df['Sale_ID'], df['Total_Amount'], marker='*', color='green')
plt.xlabel('Sale Id')
plt.ylabel('Total Amount')
plt.title('Sales')
plt.show()

df = pd.read_csv("sale_data.csv", nrows=15)
plt.plot(df['Total_Amount'], df['Quantity'])
plt.legend()
plt.title('Sales Chart')
plt.show()


df = pd.read_csv("sale_data.csv")
sales_per_region = df.groupby("Region")["Total_Amount"].sum()
print("sale:- ", sales_per_region)
colors = ['blue', 'green', 'orange', 'purple']
plt.bar(sales_per_region.index, sales_per_region.values, color=colors)
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.title("Total Sales in Region")
plt.show()


df = pd.read_csv("sale_data.csv")
colors = np.random.randint(1, 100, size=len(df['Quantity']))
plt.scatter(df['Quantity'], df['Total_Amount'], c=colors, marker='o', s=100, cmap="viridis")
plt.show()


df = pd.read_csv("sale_data.csv")
plt.hist(df['Price_per_Unit'], bins=8, edgecolor="black")
plt.show()


df = pd.read_csv("sale_data.csv")
sale_by_category = df.groupby('Category')['Total_Amount'].sum()
largest_index = sale_by_category.idxmax()
explode = [0.1 if cat == largest_index else 0 for cat in sale_by_category.index]
plt.pie(sale_by_category.values, autopct="%1.1f%%", explode=explode)
plt.show()

df = pd.read_csv("sale_data.csv")
df.boxplot(column="Total_Amount", by="Region", grid=False)
plt.title("Spread of Total_Amount across Regions")
plt.suptitle("") 
plt.xlabel("Region")
plt.ylabel("Total Amount")
plt.show()


from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("sale_data.csv")

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

regions = df["Region"].unique()
colors = plt.cm.tab10.colors
region_color_map = {region: colors[i % len(colors)] for i, region in enumerate(regions)}

for region in regions:
    subset = df[df["Region"] == region]
    ax.scatter(
        subset["Quantity"],
        subset["Price_per_Unit"],
        subset["Total_Amount"],
        color=region_color_map[region],
        label=region,
        s=50  
    )

# Labels
ax.set_xlabel("Quantity")
ax.set_ylabel("Price per Unit")
ax.set_zlabel("Total Amount")
ax.legend(title="Region")

plt.show()


df = pd.read_csv("sale_data.csv")
pivot_df = df.pivot_table(
    index="Region",
    columns="Category",
    values="Total_Amount",
    aggfunc="sum",
    fill_value=0
)

pivot_df.plot(
    kind="bar", 
    stacked=True, 
    figsize=(8,6),
    colormap="tab10"  
)

plt.title("Total Sales Amount per Region by Category")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.legend(title="Category")
plt.show()


df = pd.read_csv("sale_data.csv")
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])
monthly_avg = df.groupby(df["Sale_Date"].dt.to_period("M"))["Total_Amount"].mean()
print(monthly_avg)
# monthly_avg.plot(kind="line", marker="o", figsize=(8,5))
plt.plot(monthly_avg.index.astype(str), monthly_avg.values)
plt.show()
'''

import seaborn as sns

'''
df = pd.read_csv("sale_data.csv")
corr = df[["Quantity", "Price_per_Unit", "Total_Amount", "Discount"]].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()


df = pd.read_csv("sale_data.csv")
numeric_cols = df.select_dtypes(include="number")
corr = numeric_cols.corr()
sns.clustermap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


df = pd.read_csv("sale_data.csv")
sns.ecdfplot(data=df, x="Discount")
plt.show()


df = pd.read_csv("sale_data.csv")
numeric_cols = df.select_dtypes(include="number").columns
sns.pairplot(
    df[numeric_cols.tolist() + ["Region"]],
    hue="Region",
    kind="reg",
    diag_kind="kde",
    plot_kws={"scatter_kws": {"s": 20, "alpha": 0.6}} 
)

plt.suptitle("Pair Plot of Numerical Variables by Region", y=1.02)
plt.show()


df = pd.read_csv("sale_data.csv")

sns.jointplot(
    data=df,
    x="Quantity",
    y="Total_Amount",
    hue="Category",   
    kind="scatter",  
    height=7
)

plt.suptitle("Joint Plot of Quantity vs Total_Amount by Category", y=1.02)
plt.show()


df = pd.read_csv("sale_data.csv")
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Payment_Mode",
    y="Total_Amount",
    palette="Set2"
)

plt.title("Box Plot of Total_Amount by Payment Mode")
plt.xlabel("Payment Mode")
plt.ylabel("Total Amount")
plt.xticks(rotation=30)   # rotate if labels overlap
plt.show()


df = pd.read_csv("sale_data.csv")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    x=df["Price_per_Unit"],
    y=df["Total_Amount"],
    s=df["Quantity"] * 10,       
    c=df["Category"].astype("category").cat.codes,  
    cmap="Set2",
    alpha=0.6,
    edgecolors="w"
)

handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, df["Category"].unique(), title="Category")

plt.title("Bubble Plot: Price vs Total Amount (Bubble=Quantity, Color=Category)")
plt.xlabel("Price per Unit")
plt.ylabel("Total Amount")
plt.show()

df = pd.read_csv("sale_data.csv")
plt.figure(figsize=(10, 6))
sns.violinplot(
    x="Category",
    y="Total_Amount",
    data=df,
    inner="box",  
    palette="Set2"
)

plt.title("Violin Plot of Total Amount by Category")
plt.xlabel("Category")
plt.ylabel("Total Amount")
plt.show()
'''

df = pd.read_csv("sale_data.csv")
pivot_table = df.pivot_table(
    values="Total_Amount",
    index="Region",          
    columns="Payment_Mode",  
    aggfunc="mean"           
)

plt.figure(figsize=(10, 6))
sns.heatmap(
    pivot_table,
    annot=True,      
    fmt=".2f",       
    cmap="YlGnBu"     
)

plt.title("Average Total Amount by Region and Payment Mode")
plt.xlabel("Payment Mode")
plt.ylabel("Region")
plt.show()
