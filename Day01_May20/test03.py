class Employee:
    "二所有员工的基类"
    empCount = 0

    def ___init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)

    def displayEmployee(self):
        print("Name: ", self.name, ",Salary: ", self.salary)


emp1 = Employee("Zara", 2000)
emp2 = Employee("Manni", 5000)

# 访问属性
emp1.displayEmployee()
emp2.displayEmployee()

hasattr(emp1, "age")
getattr(emp1, "age")
setattr(emp1, "age", 8)
delattr(emp1, "age")
