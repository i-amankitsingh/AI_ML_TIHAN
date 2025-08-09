# Part A

# Python & Industry Basics ------------------------------------------------------------------------------------

'''
1. Why is Python preferred for AI/ML tasks?
Ans. Python is high level object oriented language. Python is very powerfull language, it has many powerfull library which helps in many tasks like for advanced caluclation numpy and scipy for table structure data pandas, for graphical representation matplotlib. It has very easy syntax like english langue. Here is some charactstics of python which make it best for AI/ML:
- Open source language
- Easy syntax
- Many powerfull libraries and built in functions 
- Easy to use and run and deploy
- Can perform complex task with easy syntax and minimum code

2. Name any three Python libraries useful for AI or ML.
Ans. These are three libraries are very useful for the AI or ML:-
- Numpy:- For basic and complex statestics task on big data.
- Pandas:- For tabular structure data where we deal with tabluar data
- TensorFlow:- For building machine learning models

3. What are the three core layers of an application architecture?
Ans. The main three core layers of application is:-
- Frontend:- Frontend related to the UI where user can see the data on application and intract with it's functionality.
- Backend:- Backend is used for bussiness logics. We build applications logic here like authentication, authorization, creating     user, updating and many tasks. Then connect backend with frontend.
-Database:- Database is used for storing data permanently in the memory. Using database we can store, read, update and delete the data. We connect database with backend for the storing it.


4. What does the "backend" in an application typically handle?
Ans. The backend handle the bussiness logics of the application. Here we perform core business tasks. Take a example of ecommerce website where user can see products, can order and make the payment with the platform. So these all tasks handle by the backend. To add user in the database. Fetch the all products from the database that user can see them in frontend and many other tasks.

5. Define "stateless" in the context of backend services.
Ans. The stateless means no memory of backend. It means every request which backend recieve it's new for the backend. Backend doesn't save the previous request data or memory. So whenever it's get request it's new for it and it helps to perform better task on every requests.

6. What is the purpose of APIs in application development?
Ans. As we know the three major parts of an application frontend, backend and database. So for the application it's the main task to show the data on the application frontend where user can see it and for this backend has to fetch data from the database and then pass to the frontend. So the process where data is exchange between frontend and backend is called API (Application Programming Interface). API's helps frontend and backend to easily exchange data.

7. Differentiate structured and unstructured data with one example each.
Ans. We categerized data mainly into two parts, structured and unstructured data:-
- Structured Data:- Structured data is the data which stores in the form that we can read, write and update it easily. Like a user age list in the python. user_age = [34,22,45,36]
- Unstructured Data:- Unstructured data stores in that way we can't read or write it easily and it has no strucutre like images and vidoes. 

8. Mention two reasons why Python is cloud-friendly.
Ans. Python is cloud-friendly because it has many tools where we can deploy and run it easily with the cloud plateform like google, azure SDK's. Python script is very lightweight that we can run it easily on any could plateform. And almost every cloud plateform has python support. Python can deploy and run easiy with cloud, docker container and EC2 Machin (Amazon) ect.

'''


# Numpy basics ------------------------------------------------------------------------------------

'''
1. Write code to import NumPy as np.
'''

import numpy as np


'''
2. Create a 1D array from this list: [5, 10, 15].
'''

arr = np.array([5, 10, 15])

'''
3. Generate a 2x2 array of ones using NumPy.
'''

arr = np.ones((2, 2))


''' 
4. What will np.arange(2, 10, 2) return?
'''

np_array = [2, 4, 6, 8]

'''
5. Write a NumPy command to multiply all elements of array arr by 3.
'''

arr = np.array([1,2,3,4,5])
print(arr * 3)

# Pandas Basic ------------------------------------------------------------------------------------

'''
1. Create a Pandas Series from the list [1, 3, 5].
'''

import pandas as pd
serias = pd.Series([1, 3, 5])

'''
2. Write code to create a DataFrame with columns name, age using a dictionary.
'''

data = {
    "name": ["Ankit", "Sachin", "Anuj"],
    "age": [23, 24, 21]
}

df = pd.DataFrame(data)

"""
3. What’s the difference between .iloc[2] and .loc[2]?
Ans. The iloc[2] is index position based function which find the value based on index position like .iloc[2] find the value which is on index 2. And the .loc[2] is label based indexing function which finds value based on index label like .loc[2] find the value of index which label is 2.
"""

data = {
    'name': ['Ankit', 'Sachin'],
    'age': [23, 24]
}

df = pd.DataFrame(data, index=[101, 102])

'''
In the above code .iloc[1] will print the first row which is {name: Sachin, age: 24} and .loc[102] also print the same because second row has index label 102
'''

'''
4. Write a line of code to read a CSV file into a DataFrame using Pandas.
'''

data = pd.read_csv('cardata.csv')
df = pd.DataFrame(data)


# Part B

# NumPy Tasks

# 1. Generate a 3x3 NumPy array filled with random numbers between 0 and 1.
arr = np.random.rand(3, 3)
print(arr)

# 2. From the array above, extract all elements greater than 0.5.
new_arr = arr[arr > 0.5]
print(new_arr)

# 3. Write a NumPy command to reshape a 1D array of 9 elements into 3x3.
arr = np.array([1,2,3,4,5,6,7,8,9])
new_arr = arr.reshape(3, 3)
print(new_arr)


# Pandas Selection & Cleaning

#1. Given a DataFrame df, write code to select column salary.

data = pd.read_csv('Employee-Data.csv')
df = pd.DataFrame(data)
print(df['salary'])

#2. Select the 3rd row from df using .iloc
print(df.iloc[2])

#3. Select the row with index label 104 using .loc.
print(df.loc[4]) # cause index 104 is not available so i used 4 instead of 104

#4. Write code to filter all rows where salary > 50000.
print(df[df['salary'] > 50000])

#5. How do you detect missing values in a Pandas DataFrame?
print(df.isnull())

#6. Generate a New column name score inside our DataFrame, fill it with the column’s mean.
df['score'] = df['salary'].mean()
print(df)


# Part C 
# Use Employee Data

#1. Fill missing salary values with the column mean.
df['salary'].fillna(df['salary'].mean(), inplace=True)
print(df)

#2. Convert date_joined column to datetime.
df['date_joined'] = pd.to_datetime(df['date_joined'])
print(df)

#3. Drop duplicate rows.
print(df.drop_duplicates())

#4. Rename the column date_joined to joining_date.
df = df.rename(columns={'date_joined': 'joining_date'})
print(df)

#5. Filter all employees who joined after 1st Jan 2021 using loc.
print(df[df['joining_date'] > '1-1-2021'])

#6. Show only their employee_id and department.
print(df[['employee_id', 'department']])