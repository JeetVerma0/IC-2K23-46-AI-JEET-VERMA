from collections import deque

# Graph represented using an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}


def bfs(graph, start):
    visited = set()
    queue = deque([start])

    visited.add(start)

    print("BFS Traversal:", end=" ")

    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    print()


def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)


if __name__ == "__main__":

    print("Graph Traversal using BFS and DFS")
    print("----------------------------------")

    print("\nGraph:")
    for vertex, neighbors in graph.items():
        print(vertex, "->", neighbors)

    print()

    bfs(graph, 'A')

    print("DFS Traversal:", end=" ")
    dfs(graph, 'A')
    print()
