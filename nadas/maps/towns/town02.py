from nadas.maps.town import Town


class Town02(Town):
    def __init__(self):
        spawn_dest_pairs = [
            (63, 57), (12, 96), (96, 28), (28, 78), (7, 32),
            (34, 39), (40, 36), (33, 6), (56, 35), (3, 42),
            (67, 29), (29, 97), (97, 14), (55, 62), (44, 51),
            (52, 46), (41, 2), (9, 45)
        ]

        super().__init__(
            name='Town02_Opt',
            num_vehicles=65,
            num_walkers=35,
            spawn_dest_pairs=spawn_dest_pairs,
            start_locations=None
        )
