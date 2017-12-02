# ------------------- Switch List -------------------------
# s1,s2,s3,s4,s5
# link -> cost
# s1 - s2 = 1
# s1 - s3 = 2
# s2 - s4 = 3
# s3 - s5 = 4
# s4 - s5 = 5


def data_link_cost(src, dst):

    # menggunakan dpid sebagai identitas dari switch s1 = 1, s2= 2, dst...
    # link hanya antar satu switch dengan switch lainnya
    # link_array = {([src],[dst]) = [cost]), .. }
    link_dict = {
        (1, 2): 1,
        (1, 3): 2,
        (2, 4): 3,
        (3, 5): 4,
        (4, 5): 5,
        # reverse
        (2, 1): 1,
        (3, 1): 2,
        (4, 2): 3,
        (5, 3): 4,
        (5, 4): 5,
    }

    return link_dict[src, dst]
