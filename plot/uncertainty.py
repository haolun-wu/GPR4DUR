import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(1, 50, 500)  # Generate 500 evenly spaced values between 1 and 50

# Define the function
y = np.exp(-1e-3 * (50-x))

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(x, y)

# Set the labels
plt.xlabel('x')
plt.ylabel('y')

# Set the title
plt.title('Plot of y = exp(-0.01(50 - x))')

# Display the plot
plt.show()
