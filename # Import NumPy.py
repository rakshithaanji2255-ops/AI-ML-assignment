# Import NumPy
import numpy as np

# 1. Create a NumPy array of numbers from 1 to 20
arr = np.arange(1, 23)
print("Array from 1 to 20:\n", arr)

# 2. Reshape the array into a 4x5 matrix
matrix = arr.reshape(5, 6)
print("\n4x5 Matrix:\n", matrix)

# 3. Mean, Median, Standard Deviation
mean = np.mean(arr)
median = np.median(arr)
std_dev = np.std(arr)

print("\nMean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)

# 4. Extract all even numbers
even_numbers = arr[arr % 2 == 0]
print("\nEven Numbers:\n", even_numbers)

# 5. Random 5x5 matrix and transpose
random_matrix = np.random.rand(5, 5)
transpose_matrix = random_matrix.T

print("\nRandom 5x5 Matrix:\n", random_matrix)
print("\nTranspose of Matrix:\n", transpose_matrix)
# Import Pandas
import pandas as pd

# 1. Load CSV file
df = pd.read_csv("student_data.csv")

# 2. Display first 5 and last 5 records
print("First 5 records:\n", df.head())
print("\nLast 5 records:\n", df.tail())

# 3. Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill missing values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# 4. Calculate average marks of each student
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

# 5. Student with highest average marks
top_student = df.loc[df['Average'].idxmax()]
print("\nTop Student:\n", top_student)

# 6. Students scoring more than 75 in Math
math_above_75 = df[df['Math'] > 75]
print("\nStudents scoring >75 in Math:\n", math_above_75)

# 7. Add Result column
df['Result'] = df['Average'].apply(lambda x: 'Pass' if x >= 40 else 'Fail')

print("\nUpdated DataFrame:\n", df.head())
# Import Matplotlib
import matplotlib.pyplot as plt

# 1. Bar chart: Student Names vs Average Marks
plt.figure()
plt.bar(df['Name'], df['Average'])
plt.title("Student vs Average Marks")
plt.xlabel("Student Name")
plt.ylabel("Average Marks")
plt.xticks(rotation=45)
plt.show()

# 2. Pie chart: Pass vs Fail
result_counts = df['Result'].value_counts()

plt.figure()
plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%')
plt.title("Pass vs Fail Distribution")
plt.show()

# 3. Line graph: Marks comparison
plt.figure()
plt.plot(df['Name'], df['Math'], label='Math')
plt.plot(df['Name'], df['Science'], label='Science')
plt.plot(df['Name'], df['English'], label='English')
plt.title("Marks Comparison")
plt.xlabel("Student Name")
plt.ylabel("Marks")
plt.legend()
plt.xticks(rotation=45)
plt.show()
