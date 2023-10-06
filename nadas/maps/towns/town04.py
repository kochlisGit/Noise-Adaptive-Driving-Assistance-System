from nadas.maps.town import Town


class Town04(Town):
    def __init__(self):
        spawn_dest_pairs = [
            (155, 124), (156, 125), (124, 239), (125, 238), (100, 13),
            (101, 14), (257, 268), (258, 269), (175, 92), (176, 91),
            (325, 41), (324, 42), (41, 29), (42, 30), (18, 46),
            (17, 45), (46, 304), (45, 350), (111, 275), (112, 274),
            (261, 34), (260, 33), (201, 253), (37, 252), (104, 219),
            (103, 218), (192, 242), (191, 243), (120, 246), (121, 247),
            (38, 187), (39, 84), (140, 166), (141, 166), (118, 154),
            (158, 90), (94, 144), (127, 298)
        ]

        super().__init__(
            name='Town04_Opt',
            num_vehicles=100,
            num_walkers=0,
            spawn_dest_pairs=spawn_dest_pairs,
            start_locations=None
        )
