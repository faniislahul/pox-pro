# ------------------- Switch List -------------------------
# s1,s2,s3,s4,s5
# link -> cost
# s1 - s2 = 1
# s1 - s3 = 1
# s2 - s4 = 1
# s3 - s5 = 1
# s4 - s5 = 1


def data_link_cost(src, dst):

    # link_array[src][dst] = [cost]
    link_array[1][2] = 1
    link_array[1][3] = 2
    link_array[2][4] = 3
    link_array[3][4] = 4
    link_array[4][5] = 5

    return link_array[src][dst]
