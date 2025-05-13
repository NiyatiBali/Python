# Python Section A: Short Answer & Conceptual

# 1. What is the difference between a list and a tuple in Python? Why might you choose one over the other?
# a) Lists are mutable and can be modified.Tuples are immutable and cannot be modified.
# b)	Iteration over lists is time-consuming while	Iterations over tuple is faster.
# c)	Lists are better for performing operations, such as insertion and deletion but Tuples are more suitable for accessing elements efficiently.
# d)	Lists consume more memory whereas Tuples consumes less memory.
# e)	Lists have several built-in methods while	Tuples have fewer built-in methods.
# f)	Lists are more prone to unexpected changes and errors while Tuples, being immutable are less prone to errors.

# 2. What are *args and **kwargs used for in function definitions?
#    *args in Python is used to pass a variable number of non-keyworded arguments on which we can perform tuple operations. Python **kwargs is used to pass a variable number of keyworded arguments on which we can perform dictionary operations.

# 3. Explain the difference between is and ==.
# a) == Operator: To compare objects based on their values, Python’s equality operators (==) are employed. It calls the left object’s __eq__() class method which specifies the criteria for determining equality. However, these constraints are typically written so that the equality operator == returns True if two objects, have the same value and returns False if both have different value.
# b)‘is’ Operator: Python identity operators (is, is not) are used to compare objects based on their identity. When the variables on either side of an operator point at the exact same object, the “is” operator’s evaluation is true. Otherwise, it would provide us with a false assessment.

# 4. What is a Python decorator? Provide a basic example. 
# A decorator is essentially a function that takes another function as an argument and returns a new function with enhanced functionality. Decorator takes the greet function as an argument.It returns a new function (wrapper) that first prints a message, calls greet() and then prints another message. The @decorator syntax is a shorthand for greet = decorator(greet). Example:

# A simple decorator function
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

# 5. What does the with statement do in file handling?
# When we open a file, we need to close it ourself using close(). But if something goes wrong before closing, the file might stay open, causing issues. Using with open() automatically closes the file when we’re done, even if an error happens.The with statement in Python is used for resource management and exception handling. It simplifies working with resources like files, network connections and database connections by ensuring they are properly acquired and released.

# 6. What is a generator? How is it different from a list comprehension?
# A generator expression in Python offers a memory-efficient way to create iterators. It employs a syntax similar to list comprehensions but uses parentheses instead of square brackets. Unlike list comprehensions, which construct the entire list in memory, generator expressions produce values on demand, making them suitable for large datasets or infinite sequences.
# The generator yields one item at a time and generates item only when in demand. Whereas, in a list comprehension, Python reserves memory for the whole list. Thus we can say that the generator expressions are memory efficient than the lists.

# 7. What is the difference between a POST and a GET request? When would you use each?
# a) In GET method we can not send large amount of data rather limited data of some number of characters is sent because the request parameter is appended into the URL. 	In POST method large amount of data can be sent because the request parameter is appended into the body.
# b) GET request is comparatively better than Post so it is used more than the Post request.	POST request is comparatively less better than Get method, so it is used less than the Get request.
# c) GET requests are only used to request data (not modify)	POST requests can be used to create and modify data.
# d) GET request is comparatively less secure because the data is exposed in the URL bar.	POST request is comparatively more secure because the data is not exposed in the URL bar.
# e) Request made through GET method are stored in Browser history.	Request made through POST method is not stored in Browser history.
# GET is used when you want to retrieve data and POST is used when you want to submit data. GET is often used for simple data retrieval, like fetching a web page, while POST is used for more complex operations, such as submitting a form or uploading a file.

# 8. Write a one-liner to flatten a nested list: [[1, 2], [3, 4], [5, 6]] → [1, 2, 3, 4, 5, 6].

flat_list = list(chain.from_iterable(nested_list))
print(flat_list)


# 9. How do you handle exceptions in Python? Give an example with try, except, and finally.
# These exceptions are processed using five statements. These are:
# a) try/except: catch the error and recover from exceptions hoist by programmers or Python itself.
# b) try/finally: Whether exception occurs or not, it automatically performs the clean-up action.
# c) assert: triggers an exception conditionally in the code.
# d) raise: manually triggers an exception in the code.
# e) with/as: implement context managers in older versions of Python such as - Python 2.6 & Python 3.0.

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



# 10. Explain the role of virtual environments in Python development. Why are they important when working on backend systems?
# virtual environment in Python is an isolated environment on your computer, where you can run and test your Python projects. It allows you to manage project-specific dependencies without interfering with other projects or the original Python installation. Think of a virtual environment as a separate container for each Python project. Each container: Has its own Python interpreter, its own set of installed packages, Is isolated from other virtual environments and can have different versions of the same package.
# Benefits in Backend Development:
# a) Simplified Deployment: Backend systems often need to be deployed to different servers or containers, and virtual environments make this process much easier. You can package the virtual environment along with the application code and deploy it as a single unit, ensuring that the application has the necessary dependencies. 
# b) Easier Testing: Virtual environments facilitate testing by providing a controlled environment where you can test the application's dependencies and functionality without affecting other systems. 
# c) Reduced Conflicts: In backend development, there's a greater risk of conflicts between different packages, especially when working with microservices or other complex architectures. Virtual environments help to mitigate these risks by providing a clear separation between different projects' dependencies. 

# 11. What are HTTP status codes 200, 400, and 500 typically used for? Give examples where applicable.
# a) 200 is success code (OK): This message indicates a completed request. The request could be GET, HEAD, POST or TRACE.
# b) 400 is client error code (bad request): The server says that it will not continue with the request, because of an inappropriate request (probably a syntaxes error).
# c) 500 is server error code: The most generic error possible. It doesn’t tell you anything more than the error is in the server.
