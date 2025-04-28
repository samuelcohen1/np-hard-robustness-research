from datetime import datetime
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_lin_kernighan, solve_tsp_local_search

import matplotlib.pyplot as plt
import pandas as pd


# def start_coding():
#     print("Add more Python code to this script to extend functionality!")

# def date():
#     current_datetime = datetime.now()
#     return current_datetime

def aggregate_tsp(max_points = 10, trials = 3):
    
    av_maxima = []
    av_minima = []
    av_averages = []
    av_full_tsps = []
    for n in range(2, max_points):
        maxima = []
        minima = []
        averages = []
        full_tsps = []
        for trial in range(trials):
            distances, full_distance = flex_tsp(random_points(n))
            maxima.append(max(distances))
            minima.append(min(distances))
            averages.append(np.mean(distances))
            full_tsps.append(full_distance)
        av_maxima.append(np.mean(maxima))
        av_minima.append(np.mean(minima))
        av_averages.append(np.mean(averages))
        av_full_tsps.append(np.mean(full_tsps))


    # Visualize the results in a table using pandas
    data = {
        'Number of Points': list(range(2, max_points)),
        'Maximum Flex TSP': av_maxima,        
        'Average Flex TSP': av_averages,
        'Minimum Flex TSP': av_minima,
        'Full TSPs': av_full_tsps
    }
    df = pd.DataFrame(data)
    print(df)
    # Save the DataFrame to a CSV file
    df.to_csv('tsp_results.csv', index=False)

    # # Graph the results with number of points on the x-axis and the distances on the y-axis
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['Number of Points'], df['Maxima'], label='Maxima', marker='o')
    # plt.plot(df['Number of Points'], df['Averages'], label='Averages', marker='o')
    # plt.plot(df['Number of Points'], df['Minima'], label='Minima', marker='o')
    # plt.plot(df['Number of Points'], df['Full TSPs'], label='Full TSPs', marker='o')
    # plt.title('TSP Distances vs Number of Points')
    # plt.xlabel('Number of Points')
    # plt.ylabel('Distance')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # Graph the results with number of points on the x-axis and the distances on the y-axis
    # But graph them as a percent of the full TSP
    plt.figure(figsize=(10, 6))
    plt.plot(df['Number of Points'], [x / y for x, y in zip(av_maxima, av_full_tsps)], label='Maxima', marker='o')
    plt.plot(df['Number of Points'], [x / y for x, y in zip(av_averages, av_full_tsps)], label='Averages', marker='o')
    plt.plot(df['Number of Points'], [x / y for x, y in zip(av_minima, av_full_tsps)], label='Minima', marker='o')
    plt.plot(df['Number of Points'], [x / y for x, y in zip(av_full_tsps, av_full_tsps)], label='Full TSPs', marker='o')
    plt.title('Flex TSP Distances vs Number of Points (as a percent of the full TSP)')
    plt.xlabel('Number of Points')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid()
    plt.show()

            




def flex_tsp_max(points):
    # Same as flex_tsp_plot, except we don't plot anything. Instead, we compare the distance in the original version to the max distance in the "flex" version (i.e. across all cases where one point is removed)
    # use calls to tsp() to get the distance

    # For each point in points, remove it by index, calculate the TSP, and then re-insert the point
    max_distance = 0
    for i in range(len(points)):
        new_points = np.delete(points, i, axis=0)  # Use the index `i` to remove the point
        permutation, distance = tsp(new_points, plot=False)  # Get the TSP result without showing the plot
        # Update max_distance if the current distance is greater
        if distance > max_distance:
            max_distance = distance        
    full_permutation, full_distance = tsp(points)
    return max_distance, full_distance


def flex_tsp(points):
    # Same as flex_tsp_max, but return the sorted array of distances instead of the max distance
    # For each point in points, remove it by index, calculate the TSP, and then re-insert the point
    distances = []
    for i in range(len(points)):
        new_points = np.delete(points, i, axis=0)
        # Use the index `i` to remove the point
        permutation, distance = tsp(new_points, plot=False)
        # Get the TSP result without showing the plot
        distances.append(distance)
    full_permutation, full_distance = tsp(points)
    return sorted(distances), full_distance
    


def flex_tsp_plot(points):
    plt.figure(figsize=(8, 6))  # Create a single figure for all routes
    plt.plot(points[:, 0], points[:, 1], 'o', label="Points")  # Plot the points once

    # For each point in points, remove it by index, calculate the TSP, and then re-insert the point
    for i in range(len(points)):
        new_points = np.delete(points, i, axis=0)  # Use the index `i` to remove the point
        permutation, distance = tsp(new_points, plot=False)  # Get the TSP result without showing the plot

        # Plot the route for this iteration
        for j in range(len(permutation)):
            start = new_points[permutation[j]]
            end = new_points[permutation[(j + 1) % len(permutation)]]
            # Add label only for the first blue route
            if i == 0 and j == 0:
                plt.plot([start[0], end[0]], [start[1], end[1]], color='blue', alpha=0.7, label="TSP with one point removed")
            else:
                plt.plot([start[0], end[0]], [start[1], end[1]], color='blue', alpha=0.7)

    full_permutation, full_distance = tsp(points)
    # Plot the result in red
    for j in range(len(full_permutation)):
        start = points[full_permutation[j]]
        end = points[full_permutation[(j + 1) % len(full_permutation)]]
        # Add label only for the red route
        if j == 0:
            plt.plot([start[0], end[0]], [start[1], end[1]], color='red', alpha=0.7, label="TSP with all points")
        else:
            plt.plot([start[0], end[0]], [start[1], end[1]], color='red', alpha=0.7)

    plt.title('TSP Routes for Different Point Removals')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()



def flex_tsp_plot_gradient(points, approximate=False):
    plt.figure(figsize=(8, 6))  # Create a single figure for all routes
    # plt.plot(points[:, 0], points[:, 1], 'o', label="Points")  # Plot the points once

    distances = []
    # For each point in points, remove it by index, calculate the TSP, and then re-insert the point
    for i in range(len(points)):
        print(i)
        new_points = np.delete(points, i, axis=0)  # Use the index `i` to remove the point
        permutation, distance = tsp(new_points, approximate)  # Get the TSP result without showing the plot
        distances.append(distance)
        # Plot the route for this iteration


    full_permutation, full_distance = tsp(points, approximate)
    # Plot the result in red
    for j in range(len(full_permutation)):
        start = points[full_permutation[j]]
        end = points[full_permutation[(j + 1) % len(full_permutation)]]
        # Add label only for the red route
        if j == 0:
            plt.plot([start[0], end[0]], [start[1], end[1]], color='red', alpha=0.7, label="TSP with all points")
        else:
            plt.plot([start[0], end[0]], [start[1], end[1]], color='red', alpha=0.7)


    # Color the points in the gradient based on their distance. Should be a gradient from blue to red
    # Normalize the distances for color mapping
    max_distance = max(distances)
    min_distance = min(distances)
    distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]  # Normalize the distances
    # Plot the points with gradient colors


    for i in range(len(points)):
        # Calculate the color based on the distance
        color = plt.cm.viridis(1 - distances[i])  # Normalize the distance for color mapping
        plt.plot(points[i, 0], points[i, 1], 'o', color=color, markersize=10)  # Plot the point with the gradient color
    # Add a color bar to indicate the distance scale
# Add a color bar to indicate the distance scale
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(distances), vmax=max(distances)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Change in TSP Distance with Point Removal')

    # Customize the color bar ticks and labels
    cbar.set_ticks([0, 1])  # Set ticks at the bottom (0) and top (1)
    cbar.set_ticklabels(['Min', 'Max'])  # Label the ticks as "Min" and "Max"
    

    plt.title('TSP Point Dependency')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show(block=False)



def flex_tsp_plot_dependency(points, approximate=False):
    # plt.figure(figsize=(8, 6))  # Create a single figure for all routes
    # plt.plot(points[:, 0], points[:, 1], 'o', label="Points")  # Plot the points once

    distances = []
    # For each point in points, remove it by index, calculate the TSP, and then re-insert the point
    for i in range(len(points)):
        print(i)
        new_points = np.delete(points, i, axis=0)  # Use the index `i` to remove the point
        permutation, distance = tsp(new_points, approximate)  # Get the TSP result without showing the plot
        distances.append(distance)
        # Plot the route for this iteration


    full_permutation, full_distance = tsp(points, approximate)
    print("full permutation:", full_permutation)
    # # Plot the result in red
    # for j in range(len(full_permutation)):
    #     start = points[full_permutation[j]]
    #     end = points[full_permutation[(j + 1) % len(full_permutation)]]
    #     # Add label only for the red route
    #     if j == 0:
    #         plt.plot([start[0], end[0]], [start[1], end[1]], color='red', alpha=0.7, label="TSP with all points")
    #     else:
    #         plt.plot([start[0], end[0]], [start[1], end[1]], color='red', alpha=0.7)


    # Color the points in the gradient based on their distance. Should be a gradient from blue to red
    # Normalize the distances for color mapping
    true_distances = [full_distance - d for d in distances]
    max_distance = max(distances)
    min_distance = min(distances)
    distances = [1 - (d - min_distance) / (max_distance - min_distance) for d in distances]  # Normalize the distances

    
    # Plot the distances against the distance from a point to its nearest neighbor
    triangle_changes = []
    for k in range(len(points)):
        # Get the current point C
        # Find i in the full_permutation
        # Find the index of the current point in the full_permutation
        # current_index = full_permutation.index(i)
        i = full_permutation.index(k)

        c = points[full_permutation[i]]

        # Get the adjacent points A and B in the TSP
        a = points[full_permutation[i - 1]]  # Previous point (wraps around)
        b = points[full_permutation[(i + 1) % len(points)]]  # Next point (wraps around)

        # Calculate distances AB, BC, and AC
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)

        # Calculate the change in TSP distance when C is removed
        change = ac + bc - ab
        triangle_changes.append(change)



    


    # Plot distances against nearest neighbors
    plt.figure(figsize=(8, 6))
    # plt.scatter(triangle_changes, distances,  alpha=0.7, edgecolors='k', s=100)
    plt.scatter(triangle_changes, true_distances, c=distances, cmap='viridis', alpha=0.7)
    # plt.colorbar(label='Distance')
    plt.title('TSP Distance vs Nearest Neighbor Distance')
    plt.xlabel('Neighbor triangle change in TSP distance')
    plt.ylabel('Relative change in TSP Distance with Point Removed')
    # Plot the line y = x
    plt.plot([0, max(triangle_changes)], [0, max(triangle_changes)], 'k--', label='y = x')
    plt.grid()
    plt.show()







def random_points(n=10):
    # Generate n random points in 2D space
    points = np.random.rand(n, 2) * 100
    return points

def random_tsp(n=10):
        # Define a distance matrix
    # Generate n random points in 2D space and calculate the distance matrix.
    tsp(random_points(n))


def tsp(points, approximate=False):
    # Calculate the distance matrix
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(points[i] - points[j])

    # Solve the TSP
    if approximate:
        # Use the Lin-Kernighan heuristic for an approximate solution
        permutation, distance = solve_tsp_local_search(distance_matrix)
    else:
        # Use dynamic programming for an exact solution
        # This is more computationally expensive but gives the exact solution
        # permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
        # For larger problems, consider using heuristics or approximation algorithms
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

    # if plot:
    #     # Visualize the route using matplotlib
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(points[:, 0], points[:, 1], 'o')
    #     for i in range(len(permutation)):
    #         start = points[permutation[i]]
    #         end = points[permutation[(i + 1) % len(permutation)]]
    #         plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')
    #     plt.title('Optimal TSP Route')
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Y-axis')
    #     plt.grid()
    #     plt.show()

    return permutation, distance









def main():
    # start_coding()
    # print("Current date and time:", date())
    # print("Running TSP test...")

    # print(tsp(random_points(1000), approximate=True))

    my_points = random_points(12)
    flex_tsp_plot_gradient(my_points, approximate=True)
    flex_tsp_plot_dependency(my_points, approximate=False)

    # aggregate_tsp(15, 1)

if __name__ == "__main__":
    main()