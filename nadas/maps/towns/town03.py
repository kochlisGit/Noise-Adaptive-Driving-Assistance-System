from nadas.maps.town import Town


class Town03(Town):
    def __init__(self):
        spawn_dest_pairs = [
            (191, 13), (108, 19), (262, 131), (162, 107), (30, 56),
            (45, 119), (46, 121), (239, 94), (245, 87), (89, 243),
            (220, 114), (199, 113), (38, 197), (260, 133), (1, 137),
            (258, 103), (104, 102), (34, 60), (101, 62)
        ]

        super().__init__(
            name='Town03_Opt',
            num_vehicles=65,
            num_walkers=35,
            spawn_dest_pairs=spawn_dest_pairs,
            start_locations=None
        )
