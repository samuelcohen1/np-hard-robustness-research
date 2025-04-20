from datetime import datetime
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
import matplotlib.pyplot as plt

# def start_coding():
#     print("Add more Python code to this script to extend functionality!")

# def date():
#     current_datetime = datetime.now()
#     return current_datetime

def tsp():
    # Define a distance matrix
    # Generate n random points in 2D space and calculate the distance matrix.
    n = 15  # Number of points
    points = np.random.rand(n, 2) * 100
    # print the points
    print("Points:", points)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(points[i] - points[j])
    # Solve the TSP
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    # print("Optimal route:", permutation)
    # print("Total distance:", distance)

    # Visualize the route using matplotlib
    plt.figure(figsize=(8, 6))
    plt.plot(points[:, 0], points[:, 1], 'o')
    for i in range(len(permutation)):
        start = points[permutation[i]]
        end = points[permutation[(i + 1) % len(permutation)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')
    plt.title('Optimal TSP Route')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.show()



    # distance_matrix = np.array([
    #     [0, 5, 4, 10],
    #     [5, 0, 8, 5],
    #     [4, 8, 0, 3],
    #     [10, 5, 3, 0]
    # ])
    # Solve the TSP
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    print("Optimal route:", permutation)
    print("Total distance:", distance)





def main():
    # start_coding()
    # print("Current date and time:", date())
    # print("Running TSP test...")
    tsp()

if __name__ == "__main__":
    main()