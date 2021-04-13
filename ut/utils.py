def get_subclasses(cls: type):
    frontier = cls.__subclasses__()
    visited = set()
    while frontier:
        cur = frontier.pop()
        if cur in visited:
            pass
        visited.add(cur)
        for sub in cur.__subclasses__():
            if sub not in visited:
                frontier.append(sub)
    return visited
