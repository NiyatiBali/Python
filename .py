 #Python Section A: Short Answer & Conceptual

# Q1. What is the difference between a list and a tuple in Python? Why might you choose one over the other?
# ANSWER: a) Lists are mutable and can be modified.Tuples are immutable and cannot be modified.
#         b)	Iteration over lists is time-consuming while	Iterations over tuple is faster.
#         c)	Lists are better for performing operations, such as insertion and deletion but Tuples are more suitable for accessing elements efficiently.
#         d)	Lists consume more memory whereas Tuples consumes less memory.
#         e)	Lists have several built-in methods while	Tuples have fewer built-in methods.
#         f)	Lists are more prone to unexpected changes and errors while Tuples, being immutable are less prone to errors.



# Q2. What are *args and **kwargs used for in function definitions?
# ANSWER: *args in Python is used to pass a variable number of non-keyworded arguments on which we can perform tuple operations. 
#          Python **kwargs is used to pass a variable number of keyworded arguments on which we can perform dictionary operations.



# Q3. Explain the difference between is and ==.
# ANSWER: a) == Operator: To compare objects based on their values, Python’s equality operators (==) are employed.
#                         It calls the left object’s __eq__() class method which specifies the criteria for determining equality. 
#                         However, these constraints are typically written so that the equality operator == returns True if two objects, have the same value and returns False if both have different value.
#         b)‘is’ Operator: Python identity operators (is, is not) are used to compare objects based on their identity. 
#                          When the variables on either side of an operator point at the exact same object, the “is” operator’s evaluation is true. Otherwise, it would provide us with a false assessment.



# Q4. What is a Python decorator? Provide a basic example. 
# ANSWER: A decorator is essentially a function that takes another function as an argument and returns a new function with enhanced functionality. Decorator takes the greet function as an argument.
#         It returns a new function (wrapper) that first prints a message, calls greet() and then prints another message. The @decorator syntax is a shorthand for greet = decorator(greet). Example:

# A simple decorator function
def decorator(func):
  
    def wrapper():
        print("Before calling the function.")
        func()
        print("After calling the function.")
    return wrapper

def greet():
    print("Hello, World!")

greet()



# Q5. What does the with statement do in file handling?
# ANSWER: When we open a file, we need to close it ourself using close(). But if something goes wrong before closing, the file might stay open, causing issues. 
#         Using with open() automatically closes the file when we’re done, even if an error happens.The with statement in Python is used for resource management and exception handling. 
#         It simplifies working with resources like files, network connections and database connections by ensuring they are properly acquired and released.



# Q6. What is a generator? How is it different from a list comprehension?
# ANSWER: A generator expression in Python offers a memory-efficient way to create iterators. It employs a syntax similar to list comprehensions but uses parentheses instead of square brackets. 
#         Unlike list comprehensions, which construct the entire list in memory, generator expressions produce values on demand, making them suitable for large datasets or infinite sequences.
#         The generator yields one item at a time and generates item only when in demand. Whereas, in a list comprehension, Python reserves memory for the whole list. 
#         Thus we can say that the generator expressions are memory efficient than the lists.



# Q7. What is the difference between a POST and a GET request? When would you use each?
# ANSWER: a) In GET method we can not send large amount of data rather limited data of some number of characters is sent because the request parameter is appended into the URL. 
#            In POST method large amount of data can be sent because the request parameter is appended into the body.
#         b) GET request is comparatively better than Post so it is used more than the Post request.	POST request is comparatively less better than Get method, so it is used less than the Get request.
#         c) GET requests are only used to request data (not modify)	POST requests can be used to create and modify data.
#         d) GET request is comparatively less secure because the data is exposed in the URL bar.	POST request is comparatively more secure because the data is not exposed in the URL bar.
#         e) Request made through GET method are stored in Browser history.	Request made through POST method is not stored in Browser history.
# GET is used when you want to retrieve data and POST is used when you want to submit data. 
# GET is often used for simple data retrieval, like fetching a web page, while POST is used for more complex operations, such as submitting a form or uploading a file.



# Q8. Write a one-liner to flatten a nested list: [[1, 2], [3, 4], [5, 6]] → [1, 2, 3, 4, 5, 6].
# ANSWER:
flat_list = list(chain.from_iterable(nested_list))
print(flat_list)



# Q9. How do you handle exceptions in Python? Give an example with try, except, and finally.
# ANSWER: These exceptions are processed using five statements. These are:
#          a) try/except: catch the error and recover from exceptions hoist by programmers or Python itself.
#          b) try/finally: Whether exception occurs or not, it automatically performs the clean-up action.
#          c) assert: triggers an exception conditionally in the code.
#          d) raise: manually triggers an exception in the code.
#          e) with/as: implement context managers in older versions of Python such as - Python 2.6 & Python 3.0.

  def divide(x, y): 
   
    try: 
        # Floor Division : Gives only Fractional Part as Answer 
       
        result = x // y 
    except ZeroDivisionError: 
        print("Sorry ! You are dividing by zero ") 
        
    finally:  
        # this block is always executed   
        # regardless of exception generation.  
        print('This is always executed')   



# Q10. Explain the role of virtual environments in Python development. Why are they important when working on backend systems?
# ANSWER:Virtual environment in Python is an isolated environment on your computer, where you can run and test your Python projects. 
#        It allows you to manage project-specific dependencies without interfering with other projects or the original Python installation. 
#        Think of a virtual environment as a separate container for each Python project.Each container:Has its own Python interpreter,its own set of installed packages,Is isolated from other virtual environments 
#        and can have different versions of the same package.
# Benefits in Backend Development:
#      a) Simplified Deployment: Backend systems often need to be deployed to different servers or containers, and virtual environments make this process much easier. 
#         You can package the virtual environment along with the application code and deploy it as a single unit, ensuring that the application has the necessary dependencies. 
#      b) Easier Testing: Virtual environments facilitate testing by providing a controlled environment where you can test the application's dependencies and functionality without affecting other systems. 
#      c) Reduced Conflicts: In backend development, there's a greater risk of conflicts between different packages, especially when working with microservices or other complex architectures. 
#         Virtual environments help to mitigate these risks by providing a clear separation between different projects' dependencies. 



# Q11. What are HTTP status codes 200, 400, and 500 typically used for? Give examples where applicable.
# ANSWER: a) 200 is success code (OK): This message indicates a completed request. The request could be GET, HEAD, POST or TRACE.
#         b) 400 is client error code (bad request): The server says that it will not continue with the request, because of an inappropriate request (probably a syntaxes error).
#         c) 500 is server error code: The most generic error possible. It doesn’t tell you anything more than the error is in the server.



# Section B: Coding Challenges

#Problem 1: Anagram Checker
#Write a function that checks if two strings are anagrams. Ignore case and spaces.
#def are_anagrams(str1: str, str2: str) -> bool:
#pass

def are_anagrams(str1: str, str2: str) -> bool:
    # Remove spaces and convert to lowercase
    cleaned_str1 = str1.replace(" ", "").lower()
    cleaned_str2 = str2.replace(" ", "").lower()
    
    # Check if sorted characters are the same
    return sorted(cleaned_str1) == sorted(cleaned_str2)



#Problem 2: Employee Salary Report
#Given a list of employee records as dictionaries:
#employees = [
#{"name": "Alice", "dept": "IT", "salary": 70000},
#{"name": "Bob", "dept": "HR", "salary": 50000},
#{"name": "Charlie", "dept": "IT", "salary": 80000},
#]
#Write a function to compute the average salary per department.

def average_salary_by_dept(employees):
    # Initialize a dictionary to hold the total salary and count per department
    dept_salary = {}

    for employee in employees:
        dept = employee["dept"]
        salary = employee["salary"]

        # If the department is already in the dictionary, update the total salary and count
        if dept in dept_salary:
            dept_salary[dept]["total_salary"] += salary
            dept_salary[dept]["count"] += 1
        else:
            # Otherwise, initialize the department with its salary and count
            dept_salary[dept] = {"total_salary": salary, "count": 1}

    # Now calculate the average salary per department
    avg_salaries = {}
    for dept, data in dept_salary.items():
        avg_salaries[dept] = data["total_salary"] / data["count"]

    return avg_salaries



#Problem 3: FizzBuzz Extended
#For numbers 1 to 50:
#• Print "Fizz" for multiples of 3
#• Print "Buzz" for multiples of 5
#• Print "FizzBuzz" for both
#• For prime numbers, print "Prime"

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def fizz_buzz_prime():
    for num in range(1, 51):
        # Check for Prime
        if is_prime(num):
            print(f"{num}: Prime")
        # Check for FizzBuzz
        elif num % 3 == 0 and num % 5 == 0:
            print(f"{num}: FizzBuzz")
        # Check for Fizz
        elif num % 3 == 0:
            print(f"{num}: Fizz")
        # Check for Buzz
        elif num % 5 == 0:
            print(f"{num}: Buzz")
        else:
            print(num)

# Call the function to print results
fizz_buzz_prime()



#Section C: Code Comprehension & Output Prediction

#Q1. What will be the output of the following code? Explain your answer.
#def fetch_data(items=[]):
#items.append("item")
#return items
#print(fetch_data())
#print(fetch_data())

#ANSWER: The output of the code will be:
#['item']
['item', 'item']
#Explanation:
#The function fetch_data() has a default parameter value of an empty list (items=[]). In Python, default mutable arguments, like lists or dictionaries, are shared across function calls.
#This means that the same list is used each time the function is called, rather than creating a new list each time.

#A) First call to fetch_data():
#a)Since items is not provided, it uses the default empty list ([]).
#b)"item" is appended to this list, making it ["item"].
#c)The function then returns this list: ['item'].

#B) Second call to fetch_data():
#a)Here, the same list that was modified in the first call is used. This is because default arguments are evaluated once when the function is defined, not each time the function is called.
#b)So, the list now contains ["item"] from the first call.
#c)"item" is appended again, making the list ["item", "item"].
#d)The function returns the modified list: ['item', 'item'].



#Q2. What is the output of this code, and how can it be fixed if the intention was to
#prevent shared state?
#class Counter:
#count = 0
#def increment(self):
#self.count += 1
#return self.count
#a = Counter()
#b = Counter()

#print(a.increment()) # ?
#print(b.increment()) # ?

#ANSWER: The output of this code is:
1
2
#It can be fixed using an instance variable instead of a class variable.
class Counter:
    def __init__(self):
        self.count = 0  # Instance variable

    def increment(self):
        self.count += 1
        return self.count

a = Counter()
b = Counter()

print(a.increment())  # 1
print(b.increment())  # 1
