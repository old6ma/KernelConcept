"""
Calculator Module
A simple calculator that provides basic arithmetic operations.
"""


class Calculator:
    """A calculator class that performs basic arithmetic operations."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract second number from first.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Difference of a and b
        """
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide first number by second.
        
        Args:
            a: Numerator
            b: Denominator
            
        Returns:
            Quotient of a and b
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent.
        
        Args:
            base: The base number
            exponent: The exponent
            
        Returns:
            base raised to the power of exponent
        """
        return base ** exponent
    
    def modulo(self, a: float, b: float) -> float:
        """Calculate remainder of division.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Remainder of a divided by b
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a % b
    
    def square_root(self, a: float) -> float:
        """Calculate square root of a number.
        
        Args:
            a: The number to find square root of
            
        Returns:
            Square root of a
            
        Raises:
            ValueError: If a is negative
        """
        if a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return a ** 0.5
    
    def percentage(self, part: float, whole: float) -> float:
        """Calculate percentage.
        
        Args:
            part: The part value
            whole: The whole value
            
        Returns:
            Percentage (part as percentage of whole)
            
        Raises:
            ValueError: If whole is zero
        """
        if whole == 0:
            raise ValueError("Whole cannot be zero")
        return (part / whole) * 100


def main():
    """Interactive calculator program."""
    calc = Calculator()
    
    print("=" * 50)
    print("         Simple Calculator")
    print("=" * 50)
    print("\nAvailable operations:")
    print("  +  : Addition")
    print("  -  : Subtraction")
    print("  *  : Multiplication")
    print("  /  : Division")
    print("  ** : Power")
    print("  %  : Modulo")
    print("  sqrt : Square root")
    print("  pct : Percentage")
    print("  q  : Quit")
    print("=" * 50)
    
    while True:
        print()
        operation = input("Enter operation: ").strip().lower()
        
        if operation == 'q':
            print("Goodbye!")
            break
        
        try:
            if operation == 'sqrt':
                a = float(input("Enter number: "))
                result = calc.square_root(a)
                print(f"√{a} = {result}")
            elif operation == 'pct':
                part = float(input("Enter part: "))
                whole = float(input("Enter whole: "))
                result = calc.percentage(part, whole)
                print(f"{part} is {result}% of {whole}")
            else:
                a = float(input("Enter first number: "))
                b = float(input("Enter second number: "))
                
                if operation == '+':
                    result = calc.add(a, b)
                    print(f"{a} + {b} = {result}")
                elif operation == '-':
                    result = calc.subtract(a, b)
                    print(f"{a} - {b} = {result}")
                elif operation == '*':
                    result = calc.multiply(a, b)
                    print(f"{a} × {b} = {result}")
                elif operation == '/':
                    result = calc.divide(a, b)
                    print(f"{a} ÷ {b} = {result}")
                elif operation == '**':
                    result = calc.power(a, b)
                    print(f"{a}^{b} = {result}")
                elif operation == '%':
                    result = calc.modulo(a, b)
                    print(f"{a} mod {b} = {result}")
                else:
                    print("Invalid operation. Please try again.")
                    
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
