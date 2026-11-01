from collections import deque

# Graph representing locations and connecting roads
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B', 'G'],
    'E': ['B', 'G'],
    'F': ['C', 'G'],
    'G': ['D', 'E', 'F']
}


def bfs_route(graph, start, destination):

    queue = deque([start])

    visited = {start}
    parent = {start: None}

    while queue:

        current = queue.popleft()

        if current == destination:
            break

        for neighbor in graph[current]:

            if neighbor not in visited:

                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    if destination not in parent:
        return None

    # Reconstruct route
    route = []
    current = destination

    while current is not None:
        route.append(current)
        current = parent[current]

    route.reverse()

    return route


if __name__ == "__main__":

    print("Route Planning using BFS")
    print("--------------------------")

    print("\nAvailable Locations and Roads:")

    for location, neighbors in graph.items():
        print(location, "->", neighbors)

    start = 'A'
    destination = 'G'

    print("\nStarting Location:", start)
    print("Destination:", destination)

    route = bfs_route(graph, start, destination)

    if route:

        print("\nShortest Route Found:")
        print(" -> ".join(route))

        print("Total Locations:", len(route))
        print("Total Road Segments:", len(route) - 1)

    else:
        print("\nNo route available.")
