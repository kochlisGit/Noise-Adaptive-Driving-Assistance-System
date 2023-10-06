from nadas.maps.town import Town


class Town01(Town):
    def __init__(self):
        spawn_dest_pairs = [
            (148, 199), (151, 190), (206, 216), (215, 205), (71, 97),
            (108, 217), (119, 50), (51, 130), (30, 41), (42, 31),
            (46, 64), (141, 47), (226, 187), (188, 233), (3, 12),
            (8, 2), (1, 242), (241, 0), (168, 86), (137, 167), (254, 109)
        ]

        super().__init__(
            name='Town01_Opt',
            num_vehicles=65,
            num_walkers=35,
            spawn_dest_pairs=spawn_dest_pairs,
            start_locations=None
        )
