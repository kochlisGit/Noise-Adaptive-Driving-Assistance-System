import carla
from nadas.maps.town import Town


class Town10(Town):
    def __init__(self):
        spawn_dest_pairs = [
            (6, 16), (7, 15), (83, 54), (69, 88), (19, 87),
            (20, 98), (154, 146), (131, 145), (100, 121), (120, 121),
            (123, 142), (71, 21), (112, 22), (64, 73), (128, 70),
            (78, 75), (106, 75), (93, 58), (117, 51), (32, 23),
            (116, 25), (53, 58)
        ]

        start_locations = [
            (carla.Transform(carla.Location(x=-110.764435, y=46.660076, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.642235, roll=0.000000)), 30),
            (carla.Transform(carla.Location(x=-114.232773, y=43.821014, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.642235, roll=0.000000)), 30),
            (carla.Transform(carla.Location(x=-27.160252, y=137.044220, z=0.600000), carla.Rotation(pitch=0.000000, yaw=0.352127, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-28.581730, y=140.535553, z=0.600000), carla.Rotation(pitch=0.000000, yaw=0.352127, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=45.765675, y=137.459961, z=0.600001), carla.Rotation(pitch=0.000000, yaw=0.320448, roll=0.000000)), 30),
            (carla.Transform(carla.Location(x=48.546078, y=140.975540, z=0.600001), carla.Rotation(pitch=0.000000, yaw=0.320448, roll=0.000000)), 30),
            (carla.Transform(carla.Location(x=106.377342, y=-1.649443, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-89.609253, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=109.913483, y=-6.925447, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-89.609253, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-106.686729, y=-4.847458, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-89.357758, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-103.216141, y=-2.208392, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-89.357758, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=57.568340, y=-67.849854, z=0.600000), carla.Rotation(pitch=0.000000, yaw=179.976562, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=54.469772, y=-64.348633, z=0.600000), carla.Rotation(pitch=0.000000, yaw=179.976562, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-67.045288, y=-68.693169, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-179.403244, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-64.581863, y=-65.167366, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-179.403244, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=102.566177, y=43.965668, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.390709, roll=0.000000)), 30),
            (carla.Transform(carla.Location(x=-24.336779, y=-57.785625, z=0.600000), carla.Rotation(pitch=0.000000, yaw=0.596735, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-27.800329, y=-61.284046, z=0.600000), carla.Rotation(pitch=0.000000, yaw=0.596735, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=47.557049, y=-57.225220, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-0.023438, roll=0.000000)), 25),
            (carla.Transform(carla.Location(x=44.055626, y=-60.723831, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-0.023438, roll=0.000000)), 25),
            (carla.Transform(carla.Location(x=55.542278, y=130.460068, z=0.600029), carla.Rotation(pitch=0.000000, yaw=-179.679535, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=53.122719, y=133.946625, z=0.600029), carla.Rotation(pitch=0.000000, yaw=-179.679535, roll=0.000000)), 20),
            (carla.Transform(carla.Location(x=-68.735168, y=129.303848, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-167.127060, roll=0.000000)), 25),
            (carla.Transform(carla.Location(x=-71.269684, y=132.314896, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-167.127060, roll=0.000000)), 25),
            (carla.Transform(carla.Location(x=-48.674351, y=46.955273, z=0.600000), carla.Rotation(pitch=0.000000, yaw=89.838760, roll=0.000000)), 15),
            (carla.Transform(carla.Location(x=-44.994255, y=110.955162, z=0.000000), carla.Rotation(pitch=0.000000, yaw=-90.161232, roll=0.000000)), 15),
            (carla.Transform(carla.Location(x=-25.516296, y=24.613134, z=0.600000), carla.Rotation(pitch=0.000000, yaw=0.159198, roll=0.000000)), 25),
            (carla.Transform(carla.Location(x=80.265495, y=16.907003, z=0.600000), carla.Rotation(pitch=0.000000, yaw=-179.840790, roll=0.000000)), 25),
            (carla.Transform(carla.Location(x=80.075241, y=13.406469, z=0.000000), carla.Rotation(pitch=360.000000, yaw=180.159195, roll=0.000000)), 25)
        ]

        super().__init__(
            name='Town10HD_Opt',
            num_vehicles=65,
            num_walkers=35,
            spawn_dest_pairs=spawn_dest_pairs,
            start_locations=start_locations
        )
