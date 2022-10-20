def exponential_delete(i, base=10):
    delete = []
    for j in range(i):
        power = 0
        keep = False
        while base**power < i:
            if j % base**power == 0 and j >= i - base ** (power + 1):
                keep = True
                break
            power += 1
        if not keep:
            delete.append(j)
    return delete
